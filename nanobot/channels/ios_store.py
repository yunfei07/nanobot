"""Persistent conversation store for iOS channel."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", value.lower())


def _default_name(bot_id: str) -> str:
    pretty = bot_id.replace("_", " ").replace("-", " ").strip()
    return pretty.title() if pretty else bot_id


@dataclass
class IOSBotCatalogItem:
    id: str
    name: str
    description: str
    accent_hex: str
    aliases: list[str]
    persona_prompt: str
    model: str


class IOSConversationStore:
    """Simple JSON store for iOS conversations and messages."""

    def __init__(
        self,
        workspace: Path,
        bot_profiles: dict[str, Any] | None = None,
        max_bots_per_group: int = 3,
    ) -> None:
        self.max_bots_per_group = max_bots_per_group
        self.file_path = workspace / "state" / "ios" / "conversations.v1.json"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.bots = self._build_bot_catalog(bot_profiles or {})
        self._by_alias: dict[str, str] = {}
        for bot in self.bots.values():
            self._by_alias[_normalize_token(bot.id)] = bot.id
            self._by_alias[_normalize_token(bot.name)] = bot.id
            for alias in bot.aliases:
                self._by_alias[_normalize_token(alias)] = bot.id

        self.data = self._load()
        self._ensure_minimum_data()
        self._save()

    def snapshot_for(self, sender_id: str) -> dict[str, Any]:
        conversations = [
            conversation
            for conversation in self.data["conversations"].values()
            if sender_id in conversation["member_ids"]
        ]
        conversations.sort(key=lambda c: c["updated_at"], reverse=True)

        serialized_conversations = [
            {
                "id": c["id"],
                "title": c["title"],
                "subtitle": c["subtitle"],
                "kind": c["kind"],
                "memberIDs": c["member_ids"],
                "botIDs": c["bot_ids"],
                "unreadCount": c["unread_count"],
                "updatedAt": c["updated_at"],
            }
            for c in conversations
        ]

        messages_by_conversation: dict[str, list[dict[str, Any]]] = {}
        for conversation in conversations:
            cid = conversation["id"]
            msgs = self.data["messages"].get(cid, [])
            messages_by_conversation[cid] = [
                self._serialize_message(msg, sender_id=sender_id, bot_ids=conversation["bot_ids"])
                for msg in msgs
            ]

        bots = [
            {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "accentHex": item.accent_hex,
                "aliases": item.aliases,
                "personaPrompt": item.persona_prompt,
                "model": item.model,
            }
            for item in self.bots.values()
        ]
        bots.sort(key=lambda b: b["name"])

        return {
            "currentUser": {"id": sender_id, "displayName": "You", "avatarSymbol": "person.fill"},
            "bots": bots,
            "conversations": serialized_conversations,
            "messagesByConversation": messages_by_conversation,
        }

    def create_conversation(
        self,
        *,
        title: str,
        kind: str,
        member_ids: list[str],
        bot_ids: list[str],
        created_by: str,
    ) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise ValueError("invalid_title")

        kind = kind.lower().strip()
        if kind not in {"direct", "group"}:
            raise ValueError("invalid_kind")

        normalized_members = [m.strip() for m in member_ids if m and m.strip()]
        if created_by not in normalized_members:
            normalized_members.append(created_by)
        normalized_members = sorted(set(normalized_members))
        if len(normalized_members) < 2 and kind == "group":
            raise ValueError("group_members_required")

        normalized_bots = [b.strip() for b in bot_ids if b and b.strip()]
        normalized_bots = [self.resolve_bot_token(bot) for bot in normalized_bots]
        normalized_bots = [b for b in normalized_bots if b]
        normalized_bots = list(dict.fromkeys(normalized_bots))
        if len(normalized_bots) > self.max_bots_per_group:
            raise ValueError("bot_limit_exceeded")
        if not normalized_bots:
            raise ValueError("bot_required")

        conv_id = f"c_{uuid.uuid4().hex[:10]}"
        now = _now_iso()
        conversation = {
            "id": conv_id,
            "title": title,
            "subtitle": "No messages yet",
            "kind": kind,
            "member_ids": normalized_members,
            "bot_ids": normalized_bots,
            "unread_count": 0,
            "updated_at": now,
            "created_at": now,
        }

        self.data["conversations"][conv_id] = conversation
        self.data["messages"][conv_id] = []
        self._save()
        return conversation

    def clear_history(self, conversation_id: str) -> None:
        conversation = self.data["conversations"].get(conversation_id)
        if not conversation:
            return
        conversation["subtitle"] = "No messages yet"
        conversation["unread_count"] = 0
        conversation["updated_at"] = _now_iso()
        self.data["messages"][conversation_id] = []
        self._save()

    def ensure_conversation(self, conversation_id: str, sender_id: str, default_bot_id: str | None = None) -> dict[str, Any]:
        existing = self.data["conversations"].get(conversation_id)
        if existing:
            return existing

        now = _now_iso()
        bot_id = self.resolve_bot_token(default_bot_id or "") or self._first_bot_id()
        conversation = {
            "id": conversation_id,
            "title": conversation_id,
            "subtitle": "No messages yet",
            "kind": "direct",
            "member_ids": sorted(set([sender_id, bot_id])),
            "bot_ids": [bot_id] if bot_id else [],
            "unread_count": 0,
            "updated_at": now,
            "created_at": now,
        }
        self.data["conversations"][conversation_id] = conversation
        self.data["messages"][conversation_id] = []
        self._save()
        return conversation

    def resolve_targets(self, conversation_id: str, mentions: list[str]) -> list[str]:
        conversation = self.data["conversations"].get(conversation_id)
        if not conversation:
            raise ValueError("conversation_not_found")

        conversation_bot_ids = conversation["bot_ids"]
        if not conversation_bot_ids:
            raise ValueError("bot_not_in_conversation")

        mention_ids = [
            self.resolve_bot_token(token)
            for token in mentions
        ]
        mention_ids = [token for token in mention_ids if token and token in conversation_bot_ids]

        if mention_ids:
            ordered = [bot_id for bot_id in conversation_bot_ids if bot_id in set(mention_ids)]
            return ordered

        return list(conversation_bot_ids)

    def append_user_message(self, conversation_id: str, sender_id: str, content: str) -> dict[str, Any]:
        return self._append_message(
            conversation_id=conversation_id,
            sender_id=sender_id,
            sender_role="currentUser",
            body=content,
            metadata={},
        )

    def append_bot_message(
        self,
        conversation_id: str,
        bot_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._append_message(
            conversation_id=conversation_id,
            sender_id=bot_id,
            sender_role="bot",
            body=content,
            metadata=metadata or {},
        )

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return self.data["conversations"].get(conversation_id)

    def resolve_bot_token(self, token: str) -> str | None:
        key = _normalize_token(token)
        if not key:
            return None
        return self._by_alias.get(key)

    def bot_name(self, bot_id: str) -> str:
        bot = self.bots.get(bot_id)
        return bot.name if bot else _default_name(bot_id)

    def _append_message(
        self,
        *,
        conversation_id: str,
        sender_id: str,
        sender_role: str,
        body: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        conversation = self.data["conversations"].get(conversation_id)
        if not conversation:
            raise ValueError("conversation_not_found")

        body = body.strip()
        if not body:
            raise ValueError("empty_message")

        sent_at = _now_iso()
        message = {
            "id": f"m_{uuid.uuid4().hex}",
            "conversation_id": conversation_id,
            "sender_id": sender_id,
            "sender_role": sender_role,
            "body": body,
            "sent_at": sent_at,
            "is_read": True,
            "metadata": metadata,
        }
        self.data["messages"].setdefault(conversation_id, []).append(message)

        conversation["subtitle"] = body
        conversation["updated_at"] = sent_at
        self._save()
        return message

    def _serialize_message(self, message: dict[str, Any], sender_id: str, bot_ids: list[str]) -> dict[str, Any]:
        role = message["sender_role"]
        if message["sender_id"] == sender_id:
            role = "currentUser"
        elif message["sender_id"] in bot_ids:
            role = "bot"
        elif role not in {"currentUser", "participant", "bot"}:
            role = "participant"

        return {
            "id": message["id"],
            "conversationID": message["conversation_id"],
            "senderID": message["sender_id"],
            "senderRole": role,
            "body": message["body"],
            "sentAt": message["sent_at"],
            "isRead": message["is_read"],
            "metadata": message.get("metadata", {}),
        }

    def _build_bot_catalog(self, bot_profiles: dict[str, Any]) -> dict[str, IOSBotCatalogItem]:
        if not bot_profiles:
            bot_profiles = {
                "bot_general": {},
                "bot_code": {},
                "bot_ops": {},
            }

        catalog: dict[str, IOSBotCatalogItem] = {}
        palette = ["#0CC05F", "#1D8BFF", "#FF8C1D", "#A855F7", "#EC4899", "#14B8A6"]
        for index, (bot_id, raw_profile) in enumerate(bot_profiles.items()):
            profile = raw_profile
            if hasattr(raw_profile, "model_dump"):
                profile = raw_profile.model_dump()
            if profile is None:
                profile = {}

            name = str(profile.get("name") or profile.get("display_name") or _default_name(bot_id)).strip()
            description = str(profile.get("description") or "").strip()
            accent_hex = str(profile.get("accent_hex") or palette[index % len(palette)]).strip()
            aliases = profile.get("aliases") or []
            if not isinstance(aliases, list):
                aliases = []
            aliases = [str(a).strip() for a in aliases if str(a).strip()]

            catalog[bot_id] = IOSBotCatalogItem(
                id=bot_id,
                name=name,
                description=description,
                accent_hex=accent_hex,
                aliases=aliases,
                persona_prompt=str(profile.get("persona_prompt") or "").strip(),
                model=str(profile.get("model") or "").strip(),
            )

        return catalog

    def _ensure_minimum_data(self) -> None:
        self.data.setdefault("conversations", {})
        self.data.setdefault("messages", {})

        # Keep a starter direct chat if store is empty.
        if self.data["conversations"]:
            return

        default_bot = self._first_bot_id()
        if not default_bot:
            return

        now = _now_iso()
        conv = {
            "id": "c_anna",
            "title": "Anna",
            "subtitle": "Can you help me with tonight's launch notes?",
            "kind": "direct",
            "member_ids": ["u_me", "u_anna"],
            "bot_ids": [default_bot],
            "unread_count": 1,
            "updated_at": now,
            "created_at": now,
        }
        self.data["conversations"][conv["id"]] = conv
        self.data["messages"][conv["id"]] = [
            {
                "id": f"m_{uuid.uuid4().hex}",
                "conversation_id": conv["id"],
                "sender_id": "u_anna",
                "sender_role": "participant",
                "body": "Can you help me with tonight's launch notes?",
                "sent_at": now,
                "is_read": False,
                "metadata": {},
            }
        ]

    def _first_bot_id(self) -> str | None:
        return next(iter(self.bots.keys()), None)

    def _load(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {"conversations": {}, "messages": {}}
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"conversations": {}, "messages": {}}

    def _save(self) -> None:
        temp = self.file_path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        temp.replace(self.file_path)
