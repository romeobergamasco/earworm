"""Settings from the environment, with an optional .env file (never committed)."""

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines into os.environ, without overriding what is already set."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


def require(*names: str) -> dict[str, str]:
    """The values of these variables, or a clear error naming the missing ones."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing settings: {', '.join(missing)} (see .env.example)")
    return {n: os.environ[n] for n in names}
