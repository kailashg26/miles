"""``enum.StrEnum`` shim for Python < 3.11.

``enum.StrEnum`` only exists from Python 3.11 onward, but miles targets 3.10
(``pyproject.toml`` sets ``py_version = 310``) and some runtime images (e.g. the
MI350 image) ship Python 3.10.  Import ``StrEnum`` from here instead of from
``enum`` so the same code loads on both.

On 3.11+ this re-exports the stdlib type unchanged.  On 3.10 it backports the
behaviour callers rely on: members are real ``str`` instances, ``str(member)``
returns the value (not ``"Class.MEMBER"``), and ``auto()`` yields the lowercased
member name.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` for Python 3.10."""

        __str__ = str.__str__

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()


__all__ = ["StrEnum"]
