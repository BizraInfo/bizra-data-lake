"""OBS websocket trigger adapter with graceful degradation."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class OBSTrigger:
    """Minimal OBS websocket v5 controller."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4455,
        password: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.password = password
        self._ws: Any = None
        self._connected = False

    async def connect(self) -> bool:
        try:
            import websockets
        except Exception:
            logger.warning("OBS trigger unavailable: websockets package missing")
            self._connected = False
            return False

        endpoint = f"ws://{self.host}:{self.port}"
        try:
            self._ws = await websockets.connect(endpoint)
            self._connected = True
            return True
        except Exception as exc:
            logger.warning("OBS connection failed (%s)", exc)
            self._ws = None
            self._connected = False
            return False

    async def start_recording(self) -> bool:
        return await self._request("StartRecord")

    async def stop_recording(self) -> bool:
        return await self._request("StopRecord")

    async def disconnect(self) -> None:
        if self._ws is None:
            self._connected = False
            return

        try:
            await self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None
            self._connected = False

    async def _request(self, request_type: str) -> bool:
        if not self._connected or self._ws is None:
            return False

        payload = {
            "op": 6,
            "d": {
                "requestType": request_type,
                "requestId": f"bizra-{request_type.lower()}",
            },
        }

        try:
            await self._ws.send(json.dumps(payload))
            return True
        except Exception:
            self._connected = False
            return False


__all__ = ["OBSTrigger"]
