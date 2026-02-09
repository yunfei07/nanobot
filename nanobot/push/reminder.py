"""Reminder push notification dispatcher."""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from nanobot.channels.ios_push_store import IOSPushDeviceStore


class APNsLikeClient(Protocol):
    async def send_alert(
        self,
        *,
        device_token: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Send an APNs alert notification."""


class IOSReminderNotifier:
    """Send reminder notifications to all registered iOS devices of one user."""

    _INVALID_TOKEN_REASONS = {"BadDeviceToken", "Unregistered", "DeviceTokenNotForTopic"}

    def __init__(self, *, device_store: IOSPushDeviceStore, apns_client: APNsLikeClient) -> None:
        self.device_store = device_store
        self.apns_client = apns_client

    async def notify_user_reminder(
        self,
        *,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        title: str = "提醒",
    ) -> dict[str, int]:
        tokens = self.device_store.list_tokens(user_id, platform="ios")
        if not tokens:
            logger.warning(f"APNs reminder skipped: no active iOS device for user={user_id}")
            return {"sent": 0, "failed": 0}

        sent = 0
        failed = 0
        logger.info(
            f"APNs reminder dispatch start user={user_id} devices={len(tokens)} conversation={conversation_id or '-'}"
        )
        for token in tokens:
            ok, reason = await self.apns_client.send_alert(
                device_token=token,
                title=title,
                body=message,
                data={
                    "type": "reminder",
                    "conversation_id": conversation_id or "",
                },
            )
            if ok:
                sent += 1
                logger.info(f"APNs push ok user={user_id} token={token[:8]}...{token[-6:]}")
                continue

            failed += 1
            logger.warning(
                f"APNs push failed user={user_id} token={token[:8]}...{token[-6:]} reason={reason or 'unknown'}"
            )
            if reason in self._INVALID_TOKEN_REASONS:
                self.device_store.disable_device(user_id=user_id, device_token=token)
                logger.warning(
                    f"APNs token disabled user={user_id} token={token[:8]}...{token[-6:]} reason={reason}"
                )

        logger.info(f"APNs reminder dispatch done user={user_id} sent={sent} failed={failed}")
        return {"sent": sent, "failed": failed}
