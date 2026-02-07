"""
BIZRA Channel Adapters — Platform-specific messaging gateways.

Each adapter implements the ChannelAdapter protocol to bridge
sovereign capabilities to a specific messaging platform.

Available adapters:
    - TelegramAdapter: Telegram Bot API (long polling + webhook)
"""

__all__ = ["TelegramAdapter"]


def __getattr__(name: str):
    if name == "TelegramAdapter":
        from .telegram import TelegramAdapter

        return TelegramAdapter
    raise AttributeError(f"module 'core.pat.adapters' has no attribute '{name}'")
