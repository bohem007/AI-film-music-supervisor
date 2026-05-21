"""AI Music Supervisor package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ai_music_supervisor")
except PackageNotFoundError:
    __version__ = "dev"
