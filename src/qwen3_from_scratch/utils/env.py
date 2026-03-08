from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / ".git").exists():
            return parent

    return current.parents[1]


def load_env_file(env_file_path: Optional[str] = None) -> bool:
    if env_file_path:
        env_path = Path(env_file_path)
    else:
        env_path = find_project_root() / ".env"

    if not env_path.exists():
        print(f"Warning: Environment file not found at {env_path}")
        return False
    loaded = load_dotenv(env_path, override=False)
    if not loaded:
        print(f"Warning: Failed to load environment file {env_path}")
    return loaded

