"""Shared public-repo privacy guard (pure Python, no third-party imports, CPU-only).

Two layers, applied to any text that is about to be committed:

1. ``find_local_paths``: local filesystem paths are ALWAYS rejected, for every account name including generic ones
   such as ``root``: ``C:\\...`` drive paths, ``/Users/<x>``, ``/home/<x>``, ``/root/``, ``AppData`` and
   ``.t3/worktrees``.
2. ``find_user_leaks``: the machine's account name. A distinctive name (not an OS account name) is rejected as a bare
   case-insensitive substring. A generic OS account name (``root``, ``user``, ``admin`` ...) is an ordinary English
   word in code and prose (``repo_root``, ``hecate_root``), so for those only identifying contexts are rejected:
   a path segment (``/name/``, ``\\name\\``), a credential/URL form (``name@``, ``@name``), a ``~name`` home
   shorthand, and ``user=name`` / ``username: name`` assignments.
"""
from __future__ import annotations

import re
from pathlib import Path

GENERIC_ACCOUNT_NAMES = frozenset(
    {"root", "user", "admin", "administrator", "runner", "ubuntu", "default", "owner", "agent"}
)

# Built by concatenation so that this file (and tests that import it) never contain a literal local path.
LOCAL_PATH = re.compile(
    r"\b[A-Za-z]:\\"                      # Windows drive path
    r"|/(?:Users|home)/[^/\s]+"           # macOS / Linux home directory
    r"|(?<![\w./:\-])/root(?:/|\b(?![-_.\w]))"  # the root account's home, as an absolute path
    r"|App[D]ata"
    r"|\.t3[\\/]worktrees"
)


def account_name() -> str:
    """Name of the current OS account (last component of the home directory)."""
    return Path.home().name


def is_generic_account(user: str) -> bool:
    return user.strip().lower() in GENERIC_ACCOUNT_NAMES


def find_local_paths(text: str) -> list[str]:
    """Every local-filesystem-path match in ``text`` (never user-dependent)."""
    return [m.group(0) for m in LOCAL_PATH.finditer(text)]


def _generic_user_patterns(user: str) -> list[re.Pattern[str]]:
    u = re.escape(user)
    return [
        re.compile(rf"[/\\]{u}(?=[/\\])", re.I),                       # path segment: /root/ or \root\
        re.compile(rf"(?<![\w.-]){u}@", re.I),                         # credential: root@host
        re.compile(rf"@{u}(?![\w-])", re.I),                           # mention / URL userinfo reversed
        re.compile(rf"~{u}\b", re.I),                                  # ~root home shorthand
        re.compile(rf"\b(?:user|username|login|account)\s*[=:]\s*['\"]?{u}\b", re.I),
    ]


def find_user_leaks(text: str, user: str, *, min_len: int = 4) -> list[str]:
    """Occurrences of the account name ``user`` in ``text`` that identify a person or machine.

    Distinctive names: any case-insensitive substring. Generic OS account names: identifying contexts only.
    Names shorter than ``min_len`` that are not generic are skipped (too short to be a meaningful substring test).
    """
    user = user.strip()
    if not user:
        return []
    if is_generic_account(user):
        hits: list[str] = []
        for pat in _generic_user_patterns(user):
            hits.extend(m.group(0) for m in pat.finditer(text))
        return hits
    if len(user) < min_len:
        return []
    return re.findall(re.escape(user), text, flags=re.I)


def find_leaks(text: str, user: str | None = None) -> list[str]:
    """Local paths (always) plus account-name leaks for ``user`` (default: the current account)."""
    user = account_name() if user is None else user
    return find_local_paths(text) + find_user_leaks(text, user)
