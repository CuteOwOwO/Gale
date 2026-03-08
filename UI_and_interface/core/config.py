import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def require_env(name: str) -> str:
    _load_env()
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")
