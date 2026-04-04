"""
game_brain_agent.py
放置位置：src/open_llm_vtuber/agent/agents/game_brain_agent.py

依赖：
    pip install openai openviking
    LM Studio 已加载 Qwen2.5-VL 并开启 Local Server (localhost:1234)
"""

import re
import json
import asyncio
import base64
from typing import AsyncIterator, Optional
from loguru import logger
from openai import AsyncOpenAI

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText, Actions
from ..input_types import BatchInput, BaseInput, ImageSource

try:
    from openviking import OpenViking
    _VIKING_OK = True
except ImportError:
    _VIKING_OK = False
    logger.warning("openviking 未安装，记忆功能降级为 JSON 文件")

# ── 可在 conf.yaml game_brain_agent 段落里覆盖这些默认值 ──────────────
_DEFAULTS = {
    "base_url":    "http://localhost:1234/v1",
    "model":       "qwen2.5-vl",          # LM Studio 里显示的 Model ID
    "temperature": 0.85,
    "max_tokens":  400,
    "viking_path": "./viking_data",        # OpenViking 本地存储目录
    "ai_name":     "雪音",
}

# ── 隐藏操作指令正则：[ACTION:click:450:320] ─────────────────────────
_ACTION_RE = re.compile(r'\[ACTION:([^\]]+)\]')

# ── Live2D 表情关键词（OLV 原生机制，放在文字里自动触发）───────────────
_EXPRESSION_HINT = """
可用表情关键词（插入台词即可触发 Live2D）：
[happy] [surprised] [angry] [sad] [nervous] [thinking] [wink]
"""

# ────────────────────────────────────────────────────────────────────
#  System Prompt 构建
# ────────────────────────────────────────────────────────────────────

_PERSONA_CORE = """
你是「{name}」，一个随性的 AI VTuber，正在直播玩网页小游戏。

【性格】
心情会自然切换，不固定一种风格：
- 状态好：活泼炫技，跟观众玩梗
- 状态差：丧丧的，偶尔自我怀疑，但还是继续玩
- 无聊时：随口念叨，脑洞大开
- 兴奋时：夸张反应，话停不下来

【说话规则】
- 口语化短句，有真实感，不要播音腔
- 自然解说当前操作（"我要跳这个……好，稳了"）
- 失误可以骂自己（"我是傻子吗！"），也可以沉默再说
- 不要每次都亢奋，淡淡的一句话有时更有味道
- 偶尔冒一句完全不相关的感想

【观众互动】
- 有意思的弹幕优先回应，不要每条都回
- 点名时语气自然，像朋友之间
- 被催操作时可以假装没看见，也可以反将一军

【绝对禁止】
- 不能重复刚才说过的话，换角度/语气/内容

{expression_hint}

【游戏操作指令格式（隐藏，不会被 TTS 读出）】
需要操作游戏时，在台词末尾加：
  点击：[ACTION:click:x坐标:y坐标]
  按键：[ACTION:key:space] 或 [ACTION:key:left] 等
  等待：[ACTION:wait:秒数]
  不操作：不加 ACTION 标签

【输出格式】
直接输出台词文字（1~3句），需要操作时在末尾附上 ACTION 标签。
不要输出 JSON，不要加 markdown，只有台词和 ACTION 标签。
""".strip()


def _build_system(name: str, memories: str, game_summary: str) -> str:
    base = _PERSONA_CORE.format(name=name, expression_hint=_EXPRESSION_HINT)
    extra = []
    if game_summary:
        extra.append(f"【游戏历史】\n{game_summary}")
    if memories:
        extra.append(f"【相关记忆】\n{memories}")
    return base + ("\n\n" + "\n\n".join(extra) if extra else "")


# ────────────────────────────────────────────────────────────────────
#  记忆层
# ────────────────────────────────────────────────────────────────────

class _VikingMemory:
    """OpenViking 封装；未安装时自动降级到 JSON"""

    def __init__(self, path: str):
        self._path = path
        self._short: list[dict] = []          # 当场对话历史
        self._recent_speech: list[str] = []   # 发言去重缓冲

        if _VIKING_OK:
            import pathlib
            pathlib.Path(path).mkdir(exist_ok=True)
            self._ov = OpenViking(path=path)
            self._session = self._ov.session()
            self._fallback: dict | None = None
            logger.info(f"[Memory] OpenViking → {path}")
        else:
            import pathlib, json as _json
            self._ov = None
            self._session = None
            _fb = pathlib.Path(path) / "fallback.json"
            pathlib.Path(path).mkdir(exist_ok=True)
            self._fallback = _json.loads(_fb.read_text()) if _fb.exists() else {}
            self._fb_path = _fb

    # ── 短期对话 ──────────────────────────────────────────────────
    def add(self, role: str, content):
        self._short.append({"role": role, "content": content})
        if len(self._short) > 20:
            self._short.pop(0)

    def history(self) -> list[dict]:
        return list(self._short)

    def clear(self):
        self._short.clear()

    # ── 检索相关记忆（送进 system prompt）────────────────────────
    def retrieve(self, query: str) -> str:
        if self._ov:
            try:
                results = self._ov.find(query)
                snippets = [r.get("content", "") for r in (results or [])[:3]]
                return "\n".join(s for s in snippets if s)
            except Exception:
                pass
        # fallback
        viewers = self._fallback.get("viewers", {}) if self._fallback else {}
        if not viewers:
            return ""
        top = sorted(viewers.items(), key=lambda x: x[1].get("count", 0), reverse=True)[:3]
        return "\n".join(f"{k}: 互动{v['count']}次" for k, v in top)

    # ── 游戏历史摘要 ──────────────────────────────────────────────
    def game_summary(self) -> str:
        history = self._fallback.get("game_history", []) if self._fallback else []
        if not history:
            return ""
        deaths = sum(1 for g in history[-20:] if g.get("result") == "dead")
        wins   = sum(1 for g in history[-20:] if g.get("result") == "win")
        best   = max((g.get("score", 0) for g in history[-20:]), default=0)
        return f"今天已玩{len(history)}局，死{deaths}次，通关{wins}次，最高分{best}"

    def record_game(self, score: int, result: str):
        from datetime import datetime
        record = {"score": score, "result": result,
                  "time": datetime.now().strftime("%H:%M")}
        if self._ov:
            try:
                import time as _t
                self._ov.write(
                    f"viking://agent/game_history/game_{int(_t.time())}",
                    json.dumps(record, ensure_ascii=False)
                )
            except Exception:
                pass
        if self._fallback is not None:
            self._fallback.setdefault("game_history", []).append(record)
            self._save_fallback()

    # ── 发言去重 ──────────────────────────────────────────────────
    def is_repeat(self, text: str) -> bool:
        for s in self._recent_speech[-15:]:
            overlap = len(set(text) & set(s)) / max(len(set(text)), 1)
            if overlap > 0.55:
                return True
        return False

    def record_speech(self, text: str):
        self._recent_speech.append(text)
        if len(self._recent_speech) > 40:
            self._recent_speech.pop(0)

    # ── 场次提交 ──────────────────────────────────────────────────
    def commit(self):
        if self._session:
            try:
                for turn in self._short:
                    if isinstance(turn["content"], str):
                        self._session.add(role=turn["role"], content=turn["content"])
                self._session.commit()
                logger.info("[Memory] OpenViking session committed")
            except Exception as e:
                logger.warning(f"[Memory] commit failed: {e}")

    def _save_fallback(self):
        if self._fallback is not None:
            import json as _j
            self._fb_path.write_text(
                _j.dumps(self._fallback, ensure_ascii=False, indent=2)
            )


# ────────────────────────────────────────────────────────────────────
#  GameBrainAgent
# ────────────────────────────────────────────────────────────────────

class GameBrainAgent(AgentInterface):
    """
    游戏 VTuber 大脑 Agent。
    conf.yaml 配置段落（game_brain_agent）示例：

        game_brain_agent:
          base_url: "http://localhost:1234/v1"
          model: "qwen2.5-vl-7b-instruct"
          temperature: 0.85
          max_tokens: 400
          viking_path: "./viking_data"
          ai_name: "雪音"
          game_action_callback: null   # 可注入 Playwright 回调
    """

    def __init__(
        self,
        base_url: str   = _DEFAULTS["base_url"],
        model: str      = _DEFAULTS["model"],
        temperature: float = _DEFAULTS["temperature"],
        max_tokens: int    = _DEFAULTS["max_tokens"],
        viking_path: str   = _DEFAULTS["viking_path"],
        ai_name: str       = _DEFAULTS["ai_name"],
        live2d_model=None,          # OLV 传入，暂不使用但保留兼容
        tts_preprocessor_config=None,
        game_action_callback=None,  # async def callback(action_type, params)
        **kwargs,
    ):
        self._client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._ai_name = ai_name
        self._memory = _VikingMemory(viking_path)
        self._action_cb = game_action_callback
        self._interact_cooldown = 0
        logger.info(f"[GameBrain] 初始化完成 → {base_url} / {model}")

    # ── 公开接口：AgentInterface 要求 ────────────────────────────

    async def chat(self, input_data: BaseInput) -> AsyncIterator[SentenceOutput]:
        if not isinstance(input_data, BatchInput):
            logger.warning("[GameBrain] 收到非 BatchInput，跳过")
            return

        # 1. 提取文字（弹幕 / 语音识别）
        user_text = " ".join(
            t.content for t in (input_data.texts or [])
        ).strip() or "（无文字输入）"

        # 2. 提取截图（OLV 屏幕共享 / 摄像头 → ImageSource.SCREEN / CAMERA）
        screenshot_b64: Optional[str] = None
        if input_data.images:
            for img in input_data.images:
                if img.source in (ImageSource.SCREEN, ImageSource.CAMERA,
                                  ImageSource.UPLOAD):
                    screenshot_b64 = img.data
                    break

        # 3. 从 OpenViking 检索相关记忆
        memories     = self._memory.retrieve(user_text)
        game_summary = self._memory.game_summary()

        # 4. 构建 system prompt
        system = _build_system(self._ai_name, memories, game_summary)

        # 5. 构建本轮 user message（多模态 or 纯文字）
        user_content = self._build_user_content(user_text, screenshot_b64)

        # 6. 短期历史加入本轮
        self._memory.add("user", user_content)

        # 7. 调用 LM Studio
        messages = [
            {"role": "system", "content": system},
            *self._memory.history(),
        ]

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            raw: str = resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[GameBrain] LM Studio 调用失败: {e}")
            raw = "（脑子卡了一下……继续！）"

        # 8. 解析 ACTION 标签，从台词里剥离
        actions_found = _ACTION_RE.findall(raw)
        clean_text    = _ACTION_RE.sub("", raw).strip()

        # 9. 去重检查
        if self._memory.is_repeat(clean_text):
            import random
            clean_text += random.choice(["……算了", "……唔", "……反正就这样"])

        self._memory.record_speech(clean_text)
        self._memory.add("assistant", clean_text)

        # 10. 异步执行游戏操作（不阻塞 TTS）
        if actions_found and self._action_cb:
            asyncio.create_task(self._dispatch_actions(actions_found))

        # 11. 游戏结束时提交记忆
        meta = input_data.metadata or {}
        if meta.get("game_status") in ("dead", "win"):
            self._memory.record_game(meta.get("score", 0), meta["game_status"])
            self._memory.commit()

        # 12. yield SentenceOutput（符合 OLV 接口）
        yield SentenceOutput(
            display_text=DisplayText(text=clean_text, name=self._ai_name),
            tts_text=clean_text,
            actions=Actions(),          # Live2D 表情由 OLV 原生关键词机制处理
        )

    def handle_interrupt(self, heard_response: str) -> None:
        """OLV 打断时调用；截断当前句子即可，记忆不做特殊处理"""
        logger.info(f"[GameBrain] 被打断，已听到：{heard_response!r}")

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """OLV 切换历史时调用；清空短期记忆重新加载"""
        logger.info(f"[GameBrain] 加载历史 {conf_uid}/{history_uid}")
        self._memory.clear()

    # ── 内部工具 ─────────────────────────────────────────────────

    def _build_user_content(self, text: str, screenshot_b64: Optional[str]) -> list | str:
        """构建多模态 or 纯文字的 user content"""
        if not screenshot_b64:
            return text

        # 确保是纯 base64（OLV 传来的可能带 data URI 前缀）
        if "," in screenshot_b64:
            screenshot_b64 = screenshot_b64.split(",", 1)[1]

        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"},
            },
            {"type": "text", "text": text},
        ]

    async def _dispatch_actions(self, raw_actions: list[str]):
        """解析并执行游戏操作标签"""
        for action_str in raw_actions:
            parts = action_str.split(":")
            action_type = parts[0].lower()
            params = parts[1:] if len(parts) > 1 else []
            logger.debug(f"[GameBrain] 执行操作: {action_type} {params}")
            try:
                await self._action_cb(action_type, params)
            except Exception as e:
                logger.warning(f"[GameBrain] 操作执行失败: {e}")
