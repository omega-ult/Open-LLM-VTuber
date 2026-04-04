"""
agent_factory.py 的修改说明
────────────────────────────
在原文件 else 分支之前插入以下代码块（约第 108 行附近）。
无需改动其他任何部分。
"""

# ── 插入位置：letta_agent elif 之后、else 之前 ─────────────────────

PATCH = '''
        elif conversation_agent_choice == "game_brain_agent":
            from .agents.game_brain_agent import GameBrainAgent

            s = agent_settings.get("game_brain_agent", {})
            return GameBrainAgent(
                base_url    = s.get("base_url",    "http://localhost:1234/v1"),
                model       = s.get("model",       "qwen2.5-vl"),
                temperature = s.get("temperature", 0.85),
                max_tokens  = s.get("max_tokens",  400),
                viking_path = s.get("viking_path", "./viking_data"),
                ai_name     = s.get("ai_name",     "雪音"),
                live2d_model          = live2d_model,
                tts_preprocessor_config = tts_preprocessor_config,
            )
'''

# ── 完整改动后的 agent_factory.py（仅展示关键片段）─────────────────
EXAMPLE = '''
        elif conversation_agent_choice == "letta_agent":
            ...（原有代码不动）...

        elif conversation_agent_choice == "game_brain_agent":   # ← 新增
            from .agents.game_brain_agent import GameBrainAgent

            s = agent_settings.get("game_brain_agent", {})
            return GameBrainAgent(
                base_url    = s.get("base_url",    "http://localhost:1234/v1"),
                model       = s.get("model",       "qwen2.5-vl"),
                temperature = s.get("temperature", 0.85),
                max_tokens  = s.get("max_tokens",  400),
                viking_path = s.get("viking_path", "./viking_data"),
                ai_name     = s.get("ai_name",     "雪音"),
                live2d_model            = live2d_model,
                tts_preprocessor_config = tts_preprocessor_config,
            )

        else:
            raise ValueError(f"Unsupported agent type: {conversation_agent_choice}")
'''
