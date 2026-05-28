"""Python 3.10 compatibility for ``enum.StrEnum`` (added in 3.11)."""

from enum import Enum

try:
    from enum import StrEnum
except ImportError:

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` for Python < 3.11."""
