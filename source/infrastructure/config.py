"""Deprecated: Config has been moved to source.config.

This module re-exports paths and settings for backward compatibility.
"""

from source.config import BASE_DIR, ProjectPaths, Settings, paths, settings

__all__ = ["BASE_DIR", "ProjectPaths", "Settings", "paths", "settings"]
