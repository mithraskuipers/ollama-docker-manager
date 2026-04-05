"""
Ollama Docker Manager - Configuration Manager
Handles loading and saving the JSON configuration file (ollama-config.json).
"""

import json
from pathlib import Path

from utils import OllamaConfig, Platform, PlatformDetector, ColorOutput


class ConfigManager:
    """Load, hold, and persist the application configuration."""

    def __init__(self):
        self.config = OllamaConfig()
        self.load_config()
        # Override with platform-specific defaults
        self.config.models_dir = PlatformDetector.get_models_directory()
        self.config.container_name = self._get_platform_container_name()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_platform_container_name(self) -> str:
        """Return the default container name for the current platform."""
        plat = PlatformDetector.get_platform()
        mapping = {
            Platform.WINDOWS: "ollama-win",
            Platform.LINUX:   "ollama-linux",
            Platform.WSL:     "ollama-wsl",
        }
        return mapping.get(plat, "ollama")

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_config(self) -> None:
        """Read settings from the JSON config file (if it exists)."""
        config_path = Path(self.config.config_file)
        if not config_path.exists():
            return

        try:
            with open(config_path, "r") as fh:
                data = json.load(fh)

            self.config.use_gpu              = data.get("UseGPU",               False)
            self.config.network_access       = data.get("NetworkAccess",         False)
            self.config.ollama_port          = data.get("OllamaPort",            11434)
            self.config.max_concurrent_models = data.get("MaxConcurrentModels",  1)

            # Honour an explicit container name from the file; otherwise the
            # platform default (set in __init__) takes precedence.
            saved_container = data.get("ContainerName")
            if saved_container:
                self.config.container_name = saved_container

        except Exception as exc:
            ColorOutput.warning(f"Could not load config: {exc}")

    def save_config(self) -> bool:
        """Persist current settings to the JSON config file."""
        try:
            with open(self.config.config_file, "w") as fh:
                json.dump(
                    {
                        "UseGPU":               self.config.use_gpu,
                        "NetworkAccess":        self.config.network_access,
                        "OllamaPort":           self.config.ollama_port,
                        "ContainerName":        self.config.container_name,
                        "MaxConcurrentModels":  self.config.max_concurrent_models,
                    },
                    fh,
                    indent=4,
                )
            return True
        except Exception as exc:
            ColorOutput.error(f"Could not save config: {exc}")
            return False
