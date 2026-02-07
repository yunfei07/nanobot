"""iOS channel implementation using WebSocket connections from app clients."""

import asyncio
import json
from typing import Any

import websockets
from loguru import logger
from websockets.server import WebSocketServerProtocol

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import IOSConfig


class IOSChannel(BaseChannel):
    """
    iOS channel over WebSocket.

    Protocol (JSON):
    - Client -> Server:
      - {"type":"send","senderId":"u1","chatId":"c1","content":"hi","botId":"bot_code","requestId":"..."}
      - {"type":"subscribe","chatId":"c1"} or {"type":"subscribe","chatIds":["c1","c2"]}
      - {"type":"ping"}
    - Server -> Client:
      - {"type":"hello","channel":"ios"}
      - {"type":"ack","requestId":"..."}
      - {"type":"message","chatId":"c1","content":"...","metadata":{...}}
      - {"type":"error","error":"..."}
      - {"type":"pong"}
    """

    name = "ios"

    def __init__(self, config: IOSConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: IOSConfig = config
        self._server: Any = None
        self._clients: set[WebSocketServerProtocol] = set()
        self._chat_clients: dict[str, set[WebSocketServerProtocol]] = {}

    async def start(self) -> None:
        """Start the WebSocket server for iOS clients."""
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
        """Stop the iOS WebSocket server and all client connections."""
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
        """Send an outbound message to connected iOS clients."""
        payload = {
            "type": "message",
            "channel": self.name,
            "chatId": msg.chat_id,
            "content": msg.content,
            "replyTo": msg.reply_to,
            "metadata": msg.metadata,
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
        """Handle a connected iOS WebSocket client."""
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
        """Parse and route an inbound client event."""
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

        if msg_type == "ping":
            await self._send_json(ws, {"type": "pong"})
            return

        if msg_type == "subscribe":
            self._subscribe_client(ws, data)
            await self._send_json(ws, {"type": "ack", "event": "subscribe"})
            return

        if msg_type == "send":
            await self._handle_send(ws, data)
            return

        await self._send_error(ws, f"unsupported_type:{msg_type}")

    async def _handle_send(self, ws: WebSocketServerProtocol, data: dict[str, Any]) -> None:
        """Handle a client send event and forward it to the bus."""
        if not self._is_authorized(data):
            await self._send_error(ws, "unauthorized")
            return

        sender_id = str(data.get("senderId") or data.get("sender_id") or "")
        chat_id = str(data.get("chatId") or data.get("chat_id") or "")
        content = str(data.get("content") or "")
        request_id = str(data.get("requestId") or data.get("request_id") or "")

        if not sender_id or not chat_id or not content:
            await self._send_error(ws, "missing_fields:senderId/chatId/content")
            return

        if not self.is_allowed(sender_id):
            await self._send_error(ws, "forbidden")
            return

        self._add_subscription(chat_id, ws)

        metadata: dict[str, Any] = {}
        incoming_meta = data.get("metadata")
        if isinstance(incoming_meta, dict):
            metadata.update(incoming_meta)

        bot_id = data.get("botId") or data.get("bot_id")
        if bot_id:
            metadata["bot_id"] = str(bot_id)

        conversation_type = data.get("conversationType") or data.get("conversation_type")
        if conversation_type:
            metadata["conversation_type"] = str(conversation_type)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=[],
            metadata=metadata,
        )

        await self._send_json(ws, {"type": "ack", "requestId": request_id})

    def _is_authorized(self, data: dict[str, Any]) -> bool:
        """Check optional shared-token auth for iOS clients."""
        required_token = self.config.auth_token.strip()
        if not required_token:
            return True
        provided = str(data.get("token") or "").strip()
        return provided == required_token

    def _subscribe_client(self, ws: WebSocketServerProtocol, data: dict[str, Any]) -> None:
        """Subscribe client to one or more chat IDs."""
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
        """Bind a chat_id to a connected client."""
        if chat_id not in self._chat_clients:
            self._chat_clients[chat_id] = set()
        self._chat_clients[chat_id].add(ws)

    def _remove_client(self, ws: WebSocketServerProtocol) -> None:
        """Remove disconnected client from all registries."""
        self._clients.discard(ws)

        empty_keys: list[str] = []
        for chat_id, clients in self._chat_clients.items():
            clients.discard(ws)
            if not clients:
                empty_keys.append(chat_id)

        for chat_id in empty_keys:
            self._chat_clients.pop(chat_id, None)

    async def _broadcast(self, payload: dict[str, Any], targets: list[WebSocketServerProtocol]) -> None:
        """Send payload to all target sockets, pruning stale connections."""
        to_remove: list[WebSocketServerProtocol] = []
        for ws in targets:
            try:
                await self._send_json(ws, payload)
            except Exception:
                to_remove.append(ws)

        for ws in to_remove:
            self._remove_client(ws)

    async def _send_json(self, ws: WebSocketServerProtocol, payload: dict[str, Any]) -> None:
        """Serialize and send JSON to a websocket."""
        await ws.send(json.dumps(payload, ensure_ascii=False))

    async def _send_error(self, ws: WebSocketServerProtocol, error_code: str) -> None:
        """Send structured error payload to a websocket."""
        await self._send_json(ws, {"type": "error", "error": error_code})
