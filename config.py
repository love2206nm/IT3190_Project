"""Root configuration entrypoint re-exporting source.config."""

from source.config import BASE_DIR, ProjectPaths, Settings, paths, settings

__all__ = ["BASE_DIR", "ProjectPaths", "Settings", "paths", "settings"]
