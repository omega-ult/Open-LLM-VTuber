####
# change from xTTS.py
####

import re
import requests
from loguru import logger
from .tts_interface import TTSInterface
from typing import Optional, Dict, Any


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut5",
        batch_size: str = "1",
        media_type: str = "wav",
        streaming_mode: str = "ture",
        # 新增：情绪→参考音频映射
        emotion_voice_mapping: Optional[Dict[str, Any]] = None,
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type
        self.streaming_mode = streaming_mode
        self.emotion_voice_mapping = emotion_voice_mapping or {}
        self.current_emotion = "neutral"

    def set_emotion(self, emotion: str) -> None:
        """动态设置当前情绪"""
        self.current_emotion = emotion
        logger.debug(f"[GPT-SoVITS] Emotion set to: {emotion}")

    def generate_audio(
        self,
        text: str,
        file_name_no_ext=None,
        emotion: str = None,
        vad: dict = None,
    ):
        """
        Generate audio with emotion support

        Args:
            text: Text to synthesize (may contain emotion tags like [joy])
            file_name_no_ext: Optional file name
            emotion: Emotion tag from LivOchestrator (e.g., "joy", "anger", "[joy]", "[anger]")
            vad: Optional dict {valence, arousal, dominance, intensity}.
                When provided, speed/temperature are derived continuously from VAD,
                overriding the static tts_params in emotion_voice_mapping.
        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)

        # 清除文本中的情绪标签（GPT-SoVITS 不需要）
        cleaned_text = re.sub(r"\[.*?\]", "", text)

        # 提取纯情绪名称（去除方括号）
        if emotion:
            # 支持 "[joy]" 或 "joy" 两种格式
            emotion = str(emotion).strip().strip("[]").lower()

        # 确定使用的情绪（优先级：参数 > 当前情绪 > neutral）
        active_emotion = emotion or self.current_emotion or "neutral"

        # 获取情绪对应的参考音频配置
        voice_config = self.emotion_voice_mapping.get(active_emotion, {})

        # 准备 TTS 请求参数
        data = {
            "text": cleaned_text,
            "text_language": self.text_lang,
            "refer_wav_path": voice_config.get("ref_audio_path", self.ref_audio_path),
            "prompt_text": voice_config.get("prompt_text", self.prompt_text),
            "prompt_language": self.prompt_lang,
        }

        # 添加 TTS 参数（speed, temperature 等）
        tts_params = dict(voice_config.get("tts_params", {}))

        # ── VAD 连续调制（覆盖静态 tts_params 里的 speed/temperature） ──
        # arousal 主导语速，intensity 主导随机度
        vad_log = ""
        if isinstance(vad, dict):
            arousal = float(vad.get("arousal", 0.0) or 0.0)
            intensity = float(vad.get("intensity", 0.0) or 0.0)
            arousal = max(0.0, min(1.0, arousal))
            intensity = max(0.0, min(1.0, intensity))
            vad_speed = round(0.85 + arousal * 0.40, 3)         # 0.85 → 1.25
            vad_temperature = round(0.50 + intensity * 0.50, 3)  # 0.50 → 1.00
            tts_params["speed"] = vad_speed
            tts_params["temperature"] = vad_temperature
            vad_log = f" vad_speed={vad_speed} vad_temp={vad_temperature} A={arousal:.2f} I={intensity:.2f}"

        data.update(tts_params)

        # 如果有辅助参考音频（复合情绪）
        if "aux_ref_audio_paths" in voice_config:
            data["aux_ref_audio_paths"] = voice_config["aux_ref_audio_paths"]

        # 日志输出
        log_msg = f"[GPT-SoVITS] emotion={active_emotion} text={cleaned_text[:50]}"
        if voice_config:
            log_msg += f" ref={data['refer_wav_path']}"
        log_msg += vad_log
        logger.info(log_msg)

        # Send request to the TTS API
        response = requests.post(self.api_url, json=data, timeout=120)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the audio content to a file
            with open(file_name, "wb") as audio_file:
                audio_file.write(response.content)
            return file_name
        else:
            # Handle errors or unsuccessful requests
            logger.critical(
                f"[GPT-SoVITS] Failed: status={response.status_code} "
                f"emotion={active_emotion}"
            )
            return None
