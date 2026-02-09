"""Persistent device token store for iOS push notifications."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "")


class IOSPushDeviceStore:
    """Simple JSON store for APNs device tokens."""

    def __init__(self, workspace: Path) -> None:
        self.file_path = workspace / "state" / "ios" / "devices.v1.json"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        self._save()

    def register_device(self, *, user_id: str, device_token: str, platform: str = "ios") -> dict:
        token = _normalize_token(device_token)
        if not user_id.strip() or not token:
            raise ValueError("invalid_device")

        now = _now_iso()
        for device in self.data["devices"]:
            if device["user_id"] == user_id and device["device_token"] == token:
                device["platform"] = platform
                device["enabled"] = True
                device["updated_at"] = now
                self._save()
                return device

        entry = {
            "user_id": user_id,
            "device_token": token,
            "platform": platform,
            "enabled": True,
            "updated_at": now,
        }
        self.data["devices"].append(entry)
        self._save()
        return entry

    def unregister_device(self, *, user_id: str, device_token: str) -> bool:
        token = _normalize_token(device_token)
        before = len(self.data["devices"])
        self.data["devices"] = [
            device
            for device in self.data["devices"]
            if not (device["user_id"] == user_id and device["device_token"] == token)
        ]
        removed = len(self.data["devices"]) < before
        if removed:
            self._save()
        return removed

    def disable_device(self, *, user_id: str, device_token: str) -> bool:
        token = _normalize_token(device_token)
        for device in self.data["devices"]:
            if device["user_id"] == user_id and device["device_token"] == token:
                if not device.get("enabled", True):
                    return False
                device["enabled"] = False
                device["updated_at"] = _now_iso()
                self._save()
                return True
        return False

    def list_tokens(self, user_id: str, platform: str = "ios") -> list[str]:
        return [
            str(device["device_token"])
            for device in self.data["devices"]
            if device["user_id"] == user_id
            and device.get("platform", "ios") == platform
            and bool(device.get("enabled", True))
        ]

    def _load(self) -> dict:
        if not self.file_path.exists():
            return {"version": 1, "devices": []}
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
            devices = raw.get("devices", [])
            if not isinstance(devices, list):
                devices = []
            return {"version": 1, "devices": devices}
        except Exception:
            return {"version": 1, "devices": []}

    def _save(self) -> None:
        self.file_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
