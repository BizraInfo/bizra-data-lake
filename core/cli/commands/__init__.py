"""
BIZRA CLI Commands
===================

Each command is a focused module satisfying the BaseCommand protocol.
Commands register themselves with the central CommandRegistry.
"""

from .doctor import DoctorCommand
from .dema import DemaCommand
from .identity import IdentityCommand
from .launch import LaunchCommand
from .lifecycle import ResetCommand, StartCommand, StopCommand
from .mission import MissionCommand
from .status import StatusCommand
from .version import VersionCommand
from .wallet import BriefingCommand, WalletCommand

ALL_COMMANDS = [
    DoctorCommand,
    DemaCommand,
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
