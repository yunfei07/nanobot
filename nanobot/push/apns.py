"""APNs client for sending iOS push notifications."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


class APNsClient:
    """Send alert pushes through APNs token-based auth."""

    def __init__(
        self,
        *,
        team_id: str,
        key_id: str,
        bundle_id: str,
        p8_path: str,
        use_sandbox: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.team_id = team_id.strip()
        self.key_id = key_id.strip()
        self.bundle_id = bundle_id.strip()
        self.p8_path = Path(p8_path).expanduser()
        self.use_sandbox = use_sandbox
        self.timeout_seconds = timeout_seconds
        self._cached_jwt: str | None = None
        self._cached_jwt_expire_at: int = 0

        if not self.team_id or not self.key_id or not self.bundle_id:
            raise ValueError("apns_config_incomplete")
        if not self.p8_path.exists():
            raise ValueError("apns_p8_not_found")
        self._private_key = self.p8_path.read_text(encoding="utf-8")

    async def send_alert(
        self,
        *,
        device_token: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        token = device_token.strip()
        if not token:
            return False, "invalid_device_token"

        jwt_token = self._get_provider_token()
        endpoint = "https://api.push.apple.com"
        if self.use_sandbox:
            endpoint = "https://api.sandbox.push.apple.com"

        payload: dict[str, Any] = {
            "aps": {
                "alert": {"title": title, "body": body},
                "sound": "default",
            }
        }
        if data:
            payload.update(data)

        headers = {
            "authorization": f"bearer {jwt_token}",
            "apns-topic": self.bundle_id,
            "apns-push-type": "alert",
            "apns-priority": "10",
        }
        url = f"{endpoint}/3/device/{token}"

        try:
            async with httpx.AsyncClient(http2=True, timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
        except Exception as e:
            return False, f"transport_error:{e}"

        if response.status_code == 200:
            return True, None

        reason = None
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                reason = parsed.get("reason")
        except json.JSONDecodeError:
            reason = None
        reason_text = str(reason or f"status_{response.status_code}")
        return False, reason_text

    def _get_provider_token(self) -> str:
        now = int(time.time())
        if self._cached_jwt and now < self._cached_jwt_expire_at - 30:
            return self._cached_jwt

        try:
            import jwt  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("pyjwt is required for APNs token auth") from e

        token = jwt.encode(
            {"iss": self.team_id, "iat": now},
            self._private_key,
            algorithm="ES256",
            headers={"kid": self.key_id},
        )
        self._cached_jwt = token
        self._cached_jwt_expire_at = now + (50 * 60)
        return token
