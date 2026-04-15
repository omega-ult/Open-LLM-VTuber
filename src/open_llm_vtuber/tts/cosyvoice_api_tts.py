####
# CosyVoice FastAPI TTS Engine (Zero-Shot mode)
# Calls CosyVoice's FastAPI /inference_zero_shot endpoint
# Receives PCM int16 raw bytes (22050Hz) and converts to WAV
####

import os
import re
import struct
import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:50000",
        prompt_wav: str = "",
        prompt_text: str = "",
        sample_rate: int = 22050,
        speed: float = 1.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.sample_rate = sample_rate
        self.speed = speed

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM int16 bytes to WAV format with header."""
        num_channels = 1
        sample_width = 2  # 16-bit = 2 bytes
        byte_rate = self.sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width
        data_size = len(pcm_data)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM format
            num_channels,
            self.sample_rate,
            byte_rate,
            block_align,
            sample_width * 8,
            b"data",
            data_size,
        )
        return header + pcm_data

    def generate_audio(self, text, file_name_no_ext=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, "wav")
        cleaned_text = re.sub(r"\[.*?\]", "", text)

        try:
            with open(self.prompt_wav, "rb") as f:
                response = requests.post(
                    f"{self.api_url}/inference_zero_shot",
                    data={
                        "tts_text": cleaned_text,
                        "prompt_text": self.prompt_text,
                    },
                    files={"prompt_wav": ("prompt.wav", f, "audio/wav")},
                    timeout=120,
                )
        except requests.exceptions.ConnectionError:
            logger.critical(
                f"Cannot connect to CosyVoice API at {self.api_url}. "
                "Is the CosyVoice server running?"
            )
            return None
        except FileNotFoundError:
            logger.critical(f"Prompt WAV file not found: {self.prompt_wav}")
            return None

        if response.status_code == 200:
            pcm_data = response.content
            if len(pcm_data) == 0:
                logger.critical("CosyVoice API returned empty audio data")
                return None

            wav_data = self._pcm_to_wav(pcm_data)
            with open(file_name, "wb") as audio_file:
                audio_file.write(wav_data)
            return file_name
        else:
            logger.critical(
                f"CosyVoice API error: {response.status_code} - {response.text[:200]}"
            )
            return None
