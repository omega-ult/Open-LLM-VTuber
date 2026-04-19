from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import os
import time
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)
from .conversations.tts_manager import TTSTaskManager
from .conversations.conversation_utils import (
    send_conversation_start_signals,
)
from .agent.output_types import Actions, DisplayText


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]
    request_id: Optional[str]
    target_client_uid: Optional[str]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self.allow_mic_input: bool = os.getenv("OLV_ALLOW_MIC_INPUT", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._admin_token: Optional[str] = os.getenv("OLV_ADMIN_TOKEN")
        self._last_audio_play_client_uid: Optional[str] = None
        self._last_audio_play_ts: float = 0.0

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "inject-ai-response": self._handle_inject_ai_response,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "set-mic-acceptance": self._handle_set_mic_acceptance,
            "heartbeat": self._handle_heartbeat,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # Start microphone
        if self.allow_mic_input:
            await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    def _resolve_target_client_uid(self, requested_uid: Optional[str], fallback_uid: str) -> str:
        # If requester points to itself but we know a recent playback client, prefer playback client.
        if (
            requested_uid
            and requested_uid == fallback_uid
            and self._last_audio_play_client_uid
            and self._last_audio_play_client_uid in self.client_connections
        ):
            return self._last_audio_play_client_uid

        # Prefer explicitly requested online uid
        if requested_uid and requested_uid in self.client_connections:
            return requested_uid

        # Prefer most recently active playback client
        if (
            self._last_audio_play_client_uid
            and self._last_audio_play_client_uid in self.client_connections
        ):
            return self._last_audio_play_client_uid

        # Fallback to requester if online
        if fallback_uid in self.client_connections:
            return fallback_uid

        # Last resort: any connected client
        for uid in self.client_connections.keys():
            return uid

        return fallback_uid

    def _is_admin_token_valid(self, token: Optional[str]) -> bool:
        if not self._admin_token:
            return False
        return token == self._admin_token

    def is_admin_token_valid(self, token: Optional[str]) -> bool:
        return self._is_admin_token_valid(token)

    async def _send_mic_rejected(self, websocket: WebSocket, reason: str) -> None:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "code": "MIC_INPUT_FORBIDDEN",
                    "message": reason,
                }
            )
        )

    def _clear_client_audio_buffer(self, client_uid: str) -> None:
        self.received_data_buffers[client_uid] = np.array([])

    def set_mic_acceptance(self, allow_mic_input: bool) -> None:
        self.allow_mic_input = allow_mic_input
        if not allow_mic_input:
            for uid in list(self.received_data_buffers.keys()):
                self._clear_client_audio_buffer(uid)

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_client_disconnect(self, client_uid: str):
        """Handle client disconnection"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Call context close to clean up resources (e.g., MCPClient)
        context = self.client_contexts.get(client_uid)
        if context:
            await context.close()

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def handle_disconnect(self, client_uid: str):
        """Backward-compatible disconnect entrypoint for routes."""
        await self.handle_client_disconnect(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        if not self.allow_mic_input:
            self._clear_client_audio_buffer(client_uid)
            await self._send_mic_rejected(websocket, "Mic input is currently disabled")
            return

        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        if not self.allow_mic_input:
            self._clear_client_audio_buffer(client_uid)
            await self._send_mic_rejected(websocket, "Mic input is currently disabled")
            return

        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        msg_type = data.get("type", "")
        if not self.allow_mic_input and msg_type in {
            "mic-audio-end",
            "text-input",
            "ai-speak-signal",
        }:
            self._clear_client_audio_buffer(client_uid)
            await self._send_mic_rejected(
                websocket,
                "Direct user input is currently disabled; inject input only",
            )
            return

        await handle_conversation_trigger(
            msg_type=msg_type,
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )

    async def _handle_inject_ai_response(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle pre-processed AI responses from external orchestrator (LivOchestrator).
        Bypasses the LLM agent and goes directly to TTS + Live2D.
        Broadcasts to ALL connected clients (OBS browser, admin page, etc.).

        Requires admin_token for authorization when OLV_ADMIN_TOKEN is set.
        """
        # Token validation for inject-ai-response
        token = data.get("admin_token")
        if not self._is_admin_token_valid(token):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "code": "UNAUTHORIZED",
                        "message": "Invalid or missing admin token for inject-ai-response",
                    }
                )
            )
            logger.warning(f"inject-ai-response rejected: invalid token from client {client_uid}")
            return

        context = self.client_contexts.get(client_uid)
        if not context:
            logger.warning(f"inject-ai-response: no context for client {client_uid}")
            return

        text = data.get("text", "")
        motion = data.get("motion")  # Motion name from LivOchestrator
        request_id = str(data.get("request_id") or f"inject-{client_uid}-{int(asyncio.get_running_loop().time() * 1000)}")
        requested_target_uid = data.get("target_client_uid")
        target_client_uid = self._resolve_target_client_uid(
            str(requested_target_uid) if requested_target_uid else None,
            fallback_uid=client_uid,
        )
        logger.info(
            f"inject-ai-response routing: request_id={request_id} "
            f"requested_target={requested_target_uid or '-'} resolved_target={target_client_uid}"
        )
        if not text:
            logger.warning("inject-ai-response: empty text")
            return

        tts_manager = TTSTaskManager()

        # Build a broadcast sender that sends to all connected clients
        async def broadcast_send(message: str):
            dead = []
            for uid, ws in list(self.client_connections.items()):
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(uid)
            for uid in dead:
                self.client_connections.pop(uid, None)

        completion_status = "completed"
        completion_source = "frontend-playback-complete"
        completion_reason = ""

        try:
            await broadcast_send(
                json.dumps(
                    {
                        "type": "control",
                        "text": "conversation-chain-start",
                        "request_id": request_id,
                        "target_client_uid": target_client_uid,
                    }
                )
            )

            # Extract emotion for Live2D
            expression_list = context.live2d_model.extract_emotion(text)
            clean_text = context.live2d_model.remove_emotion_keywords(text)

            # 提取情绪标签（优先级：emotion 字段 > 文本解析 > neutral）
            emotion_from_field = data.get("emotion")  # 从 WebSocket payload 获取
            emotion_tag = emotion_from_field or self._extract_emotion_from_text(text)

            # Build actions for Live2D
            actions = Actions(expressions=expression_list if expression_list else None, motion=motion)

            # Build display text
            display_text = DisplayText(
                text=clean_text,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )

            # Queue TTS — broadcast to all clients
            await tts_manager.speak(
                tts_text=clean_text,
                display_text=display_text,
                actions=actions,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=broadcast_send,
                request_id=request_id,
                target_client_uid=target_client_uid,
                emotion=emotion_tag,  # 传递情绪到 TTS
            )

            # Wait for TTS completion
            if tts_manager.task_list:
                await asyncio.gather(*tts_manager.task_list)
                await broadcast_send(
                    json.dumps(
                        {
                            "type": "backend-synth-complete",
                            "request_id": request_id,
                            "target_client_uid": target_client_uid,
                        }
                    )
                )

            await broadcast_send(json.dumps({"type": "force-new-message"}))

            chain_end_msg = {
                "type": "control",
                "text": "conversation-chain-end",
                "request_id": request_id,
                "target_client_uid": target_client_uid,
            }
            await broadcast_send(json.dumps(chain_end_msg))

            response = await message_handler.wait_for_response(
                target_client_uid,
                "frontend-playback-complete",
                request_id=request_id,
                timeout=45.0,
            )
            if not response:
                completion_status = "timeout"
                completion_source = "server-timeout"
                completion_reason = "frontend-playback-complete timeout"

        except Exception as e:
            completion_status = "error"
            completion_source = "server-exception"
            completion_reason = str(e)
            logger.error(f"Error in inject-ai-response: {e}")
        finally:
            completion_payload = {
                "type": "inject-ai-response-complete",
                "request_id": request_id,
                "target_client_uid": target_client_uid,
                "status": completion_status,
                "source": completion_source,
            }
            if completion_reason:
                completion_payload["reason"] = completion_reason
            try:
                await websocket.send_text(json.dumps(completion_payload))
            except Exception as send_err:
                logger.warning(f"inject-ai-response-complete send failed: {send_err}")
            tts_manager.clear()

    def _extract_emotion_from_text(self, text: str) -> str:
        """从文本中提取情绪标签"""
        import re
        match = re.search(r"\[(\w+)\]", text)
        return match.group(1).lower() if match else "neutral"

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        self._last_audio_play_client_uid = client_uid
        self._last_audio_play_ts = time.time()
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

    async def _handle_set_mic_acceptance(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        payload = data.get("data") or {}
        token = payload.get("admin_token")
        if not self._is_admin_token_valid(token):
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "code": "UNAUTHORIZED",
                        "message": "Invalid admin token",
                    }
                )
            )
            return

        allow_mic_input = bool(payload.get("allow_mic_input", True))
        self.set_mic_acceptance(allow_mic_input)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-mic-acceptance-result",
                    "data": {
                        "ok": True,
                        "allow_mic_input": self.allow_mic_input,
                    },
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")
