"""
BIZRA CLI Commands
===================

Each command is a focused module satisfying the BaseCommand protocol.
Commands register themselves with the central CommandRegistry.
"""

from .doctor import DoctorCommand
from .identity import IdentityCommand
from .launch import LaunchCommand
from .lifecycle import StartCommand, StopCommand, ResetCommand
from .mission import MissionCommand
from .status import StatusCommand
from .version import VersionCommand
from .wallet import WalletCommand, BriefingCommand

ALL_COMMANDS = [
    DoctorCommand,
    IdentityCommand,
    LaunchCommand,
    StartCommand,
    StopCommand,
    ResetCommand,
    MissionCommand,
    StatusCommand,
    VersionCommand,
    WalletCommand,
    BriefingCommand,
]

__all__ = ["ALL_COMMANDS"]
