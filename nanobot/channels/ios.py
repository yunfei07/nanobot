"""iOS channel implementation using WebSocket connections from app clients."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import websockets
from loguru import logger
from websockets.server import WebSocketServerProtocol

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.ios_store import IOSConversationStore
from nanobot.config.schema import IOSConfig


class IOSChannel(BaseChannel):
    """iOS channel over WebSocket with persistent conversations."""

    name = "ios"

    def __init__(
        self,
        config: IOSConfig,
        bus: MessageBus,
        *,
        workspace: Path,
        bot_profiles: dict[str, Any] | None = None,
    ):
        super().__init__(config, bus)
        self.config: IOSConfig = config
        self._server: Any = None
        self._clients: set[WebSocketServerProtocol] = set()
        self._chat_clients: dict[str, set[WebSocketServerProtocol]] = {}
        self.store = IOSConversationStore(workspace=workspace, bot_profiles=bot_profiles)

    async def start(self) -> None:
        self._running = True
        self._server = await websockets.serve(
            self._on_client,
            self.config.host,
            self.config.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=2 * 1024 * 1024,
        )
        logger.info(f"iOS channel listening on ws://{self.config.host}:{self.config.port}")

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False

        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self._clients.clear()
        self._chat_clients.clear()

    async def send(self, msg: OutboundMessage) -> None:
        metadata = dict(msg.metadata or {})
        bot_id = str(metadata.get("bot_id") or "")
        if bot_id:
            try:
                stored = self.store.append_bot_message(
                    conversation_id=msg.chat_id,
                    bot_id=bot_id,
                    content=msg.content,
                    metadata=metadata,
                )
                metadata["message_id"] = stored["id"]
            except Exception:
                pass

        payload = {
            "type": "message",
            "channel": self.name,
            "chatId": msg.chat_id,
            "chat_id": msg.chat_id,
            "messageId": metadata.get("message_id"),
            "senderId": bot_id or "bot",
            "senderDisplayName": self.store.bot_name(bot_id or "bot"),
            "content": msg.content,
            "replyTo": msg.reply_to,
            "replyToClientMessageId": metadata.get("client_request_id"),
            "metadata": metadata,
        }

        targets = self._chat_clients.get(msg.chat_id)
        if targets:
            await self._broadcast(payload, list(targets))
            return

        if self._clients:
            await self._broadcast(payload, list(self._clients))
            return

        logger.warning(f"No iOS clients connected for chat_id={msg.chat_id}")

    async def _on_client(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        await self._send_json(ws, {"type": "hello", "channel": self.name})

        try:
            async for raw in ws:
                await self._handle_client_message(ws, raw)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.warning(f"iOS client error: {e}")
        finally:
            self._remove_client(ws)

    async def _handle_client_message(self, ws: WebSocketServerProtocol, raw: Any) -> None:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
        except Exception:
            await self._send_error(ws, "invalid_json")
            return

        if not isinstance(data, dict):
            await self._send_error(ws, "invalid_payload")
            return

        msg_type = str(data.get("type", "")).lower()
        request_id = str(data.get("requestId") or data.get("request_id") or "")

        if msg_type == "ping":
            await self._send_json(ws, {"type": "pong"})
            return

        if msg_type == "subscribe":
            self._subscribe_client(ws, data)
            await self._send_json(ws, {"type": "ack", "event": "subscribe", "requestId": request_id})
            return

        if msg_type == "bootstrap":
            await self._handle_bootstrap(ws, data, request_id=request_id)
            return

        if msg_type in {"createconversation", "create_conversation"}:
            await self._handle_create_conversation(ws, data, request_id=request_id)
            return

        if msg_type in {"clearhistory", "clear_history"}:
            await self._handle_clear_history(ws, data, request_id=request_id)
            return

        if msg_type == "send":
            await self._handle_send(ws, data, request_id=request_id)
            return

        await self._send_error(ws, f"unsupported_type:{msg_type}", request_id=request_id)

    async def _handle_bootstrap(self, ws: WebSocketServerProtocol, data: dict[str, Any], request_id: str) -> None:
        if not self._is_authorized(data):
            await self._send_error(ws, "unauthorized", request_id=request_id)
            return

        sender_id = str(data.get("senderId") or data.get("sender_id") or "")
        if not sender_id:
            await self._send_error(ws, "missing_fields:senderId", request_id=request_id)
            return
        if not self.is_allowed(sender_id):
            await self._send_error(ws, "forbidden", request_id=request_id)
            return

        snapshot = self.store.snapshot_for(sender_id)
        await self._send_json(
            ws,
            {
                "type": "bootstrap",
                "requestId": request_id,
                **snapshot,
            },
        )

    async def _handle_create_conversation(self, ws: WebSocketServerProtocol, data: dict[str, Any], request_id: str) -> None:
        if not self._is_authorized(data):
            await self._send_error(ws, "unauthorized", request_id=request_id)
            return

        sender_id = str(data.get("senderId") or data.get("sender_id") or "")
        if not sender_id:
            await self._send_error(ws, "missing_fields:senderId", request_id=request_id)
            return
        if not self.is_allowed(sender_id):
            await self._send_error(ws, "forbidden", request_id=request_id)
            return

        title = str(data.get("title") or "")
        kind = str(data.get("kind") or "group")
        member_ids = self._extract_str_list(data.get("memberIds") or data.get("member_ids") or [])
        bot_ids = self._extract_str_list(data.get("botIds") or data.get("bot_ids") or [])

        try:
            conversation = self.store.create_conversation(
                title=title,
                kind=kind,
                member_ids=member_ids,
                bot_ids=bot_ids,
                created_by=sender_id,
            )
        except ValueError as e:
            await self._send_error(ws, str(e), request_id=request_id)
            return

        self._add_subscription(conversation["id"], ws)
        await self._send_json(
            ws,
            {
                "type": "conversation_created",
                "requestId": request_id,
                "conversation": {
                    "id": conversation["id"],
                    "title": conversation["title"],
                    "subtitle": conversation["subtitle"],
                    "kind": conversation["kind"],
                    "memberIDs": conversation["member_ids"],
                    "botIDs": conversation["bot_ids"],
                    "unreadCount": conversation["unread_count"],
                    "updatedAt": conversation["updated_at"],
                },
            },
        )

    async def _handle_clear_history(self, ws: WebSocketServerProtocol, data: dict[str, Any], request_id: str) -> None:
        if not self._is_authorized(data):
            await self._send_error(ws, "unauthorized", request_id=request_id)
            return

        sender_id = str(data.get("senderId") or data.get("sender_id") or "")
        chat_id = str(data.get("chatId") or data.get("chat_id") or "")
        if not sender_id or not chat_id:
            await self._send_error(ws, "missing_fields:senderId/chatId", request_id=request_id)
            return
        if not self.is_allowed(sender_id):
            await self._send_error(ws, "forbidden", request_id=request_id)
            return

        conversation = self.store.get_conversation(chat_id)
        if not conversation:
            await self._send_error(ws, "conversation_not_found", request_id=request_id)
            return
        if sender_id not in conversation["member_ids"]:
            await self._send_error(ws, "forbidden", request_id=request_id)
            return

        self.store.clear_history(chat_id)
        await self._send_json(ws, {"type": "ack", "event": "clear_history", "requestId": request_id})

    async def _handle_send(self, ws: WebSocketServerProtocol, data: dict[str, Any], request_id: str) -> None:
        if not self._is_authorized(data):
            await self._send_error(ws, "unauthorized", request_id=request_id)
            return

        sender_id = str(data.get("senderId") or data.get("sender_id") or "")
        chat_id = str(data.get("chatId") or data.get("chat_id") or "")
        content = str(data.get("content") or "")

        if not sender_id or not chat_id or not content:
            await self._send_error(ws, "missing_fields:senderId/chatId/content", request_id=request_id)
            return

        if not self.is_allowed(sender_id):
            await self._send_error(ws, "forbidden", request_id=request_id)
            return

        default_bot = str(data.get("botId") or data.get("bot_id") or "")
        conversation = self.store.ensure_conversation(chat_id, sender_id, default_bot_id=default_bot)
        if sender_id not in conversation["member_ids"]:
            await self._send_error(ws, "forbidden", request_id=request_id)
            return

        self._add_subscription(chat_id, ws)

        raw_mentions = self._extract_str_list(data.get("mentions") or data.get("mention_ids") or [])
        mentions = [self._clean_mention(token) for token in raw_mentions]
        mentions = [token for token in mentions if token]
        if not mentions:
            mentions = self._extract_mentions_from_content(content)
        try:
            target_bot_ids = self.store.resolve_targets(chat_id, mentions)
        except ValueError as e:
            await self._send_error(ws, str(e), request_id=request_id)
            return

        try:
            self.store.append_user_message(chat_id, sender_id, content)
        except ValueError as e:
            await self._send_error(ws, str(e), request_id=request_id)
            return

        incoming_meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        conversation_type = data.get("conversationType") or data.get("conversation_type") or conversation["kind"]

        for bot_id in target_bot_ids:
            metadata: dict[str, Any] = dict(incoming_meta)
            metadata["bot_id"] = bot_id
            metadata["client_request_id"] = request_id
            metadata["conversation_type"] = str(conversation_type)
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=[],
                metadata=metadata,
            )

        await self._send_json(
            ws,
            {
                "type": "ack",
                "event": "send",
                "requestId": request_id,
                "targetBotIds": target_bot_ids,
            },
        )

    def _is_authorized(self, data: dict[str, Any]) -> bool:
        required_token = self.config.auth_token.strip()
        if not required_token:
            return True
        provided = str(data.get("token") or "").strip()
        return provided == required_token

    def _subscribe_client(self, ws: WebSocketServerProtocol, data: dict[str, Any]) -> None:
        chat_ids: list[str] = []
        single = data.get("chatId") or data.get("chat_id")
        if isinstance(single, str) and single:
            chat_ids.append(single)
        many = data.get("chatIds") or data.get("chat_ids")
        if isinstance(many, list):
            for item in many:
                if isinstance(item, str) and item:
                    chat_ids.append(item)
        for chat_id in chat_ids:
            self._add_subscription(chat_id, ws)

    def _add_subscription(self, chat_id: str, ws: WebSocketServerProtocol) -> None:
        if chat_id not in self._chat_clients:
            self._chat_clients[chat_id] = set()
        self._chat_clients[chat_id].add(ws)

    def _remove_client(self, ws: WebSocketServerProtocol) -> None:
        self._clients.discard(ws)
        empty_keys: list[str] = []
        for chat_id, clients in self._chat_clients.items():
            clients.discard(ws)
            if not clients:
                empty_keys.append(chat_id)
        for chat_id in empty_keys:
            self._chat_clients.pop(chat_id, None)

    async def _broadcast(self, payload: dict[str, Any], targets: list[WebSocketServerProtocol]) -> None:
        to_remove: list[WebSocketServerProtocol] = []
        for ws in targets:
            try:
                await self._send_json(ws, payload)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self._remove_client(ws)

    async def _send_json(self, ws: WebSocketServerProtocol, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, ensure_ascii=False))

    async def _send_error(self, ws: WebSocketServerProtocol, error_code: str, request_id: str = "") -> None:
        payload: dict[str, Any] = {"type": "error", "error": error_code}
        if request_id:
            payload["requestId"] = request_id
        await self._send_json(ws, payload)

    @staticmethod
    def _extract_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    @staticmethod
    def _clean_mention(token: str) -> str:
        token = token.strip()
        if token.startswith("@"):
            token = token[1:]
        return re.sub(r"[^\w-]+", "", token).lower()

    @classmethod
    def _extract_mentions_from_content(cls, text: str) -> list[str]:
        found = re.findall(r"@([\w\-\u4e00-\u9fff]+)", text)
        cleaned: list[str] = []
        for token in found:
            normalized = cls._clean_mention(token)
            if normalized:
                cleaned.append(normalized)
        return cleaned
