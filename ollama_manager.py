#!/usr/bin/env python3
"""
Ollama Docker Manager - Main Entry Point
Wires together all modules and runs the main menu loop.

Module layout
─────────────
  ollama_manager.py     ← you are here  (orchestrator + entry point)
  config_manager.py     ← load / save ollama-config.json
  menu_handler.py       ← every menu-action method
  system_setup.py       ← Linux/WSL requirements screen & installers
  docker_manager.py     ← Docker / container / model operations
  turboquant_manager.py ← TurboQuant server management
  utils.py              ← shared data-classes, colors, platform detection

Dependencies
────────────
  Python 3.7+, Docker, requests (auto-installed into shared venv)
"""

import os
import sys

# Platform-specific stdin imports used for chat interruption
if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty
    import select

from utils import Platform, Colors, ColorOutput, PlatformDetector, VenvManager
from docker_manager import DockerManager
from turboquant_manager import TurboQuantManager
from config_manager import ConfigManager
from menu_handler import MenuHandler
from system_setup import SystemSetup


class OllamaManager:
    """
    Top-level application class.

    Initialises all sub-managers, shows the main menu, and dispatches
    user input to the appropriate handler.
    """

    def __init__(self):
        self.cfg        = ConfigManager()
        self.docker     = DockerManager(self.cfg.config)
        self.turboquant = TurboQuantManager()
        self.platform   = PlatformDetector.get_platform()

        self.menu  = MenuHandler(self.docker, self.cfg, self.turboquant, self.platform)
        self.setup = SystemSetup(self.platform, self.turboquant)

    # ── Main menu display ──────────────────────────────────────────────────────

    def show_main_menu(self) -> None:
        """Render the main menu with live status indicators."""
        ColorOutput.print("=" * 60, Colors.CYAN, bold=True)
        ColorOutput.print("       🦙 OLLAMA DOCKER MANAGER - Cross Platform", Colors.CYAN, bold=True)
        ColorOutput.print("=" * 60, Colors.CYAN, bold=True)
        print()

        ColorOutput.print(f"Platform:  {self.platform.value}", Colors.GRAY)
        ColorOutput.print(
            f"Container: {self.cfg.config.container_name}  |  Port: {self.cfg.config.ollama_port}",
            Colors.GRAY,
        )
        print()

        # ── Status line ──────────────────────────────────────────────────────
        image_ok = self.docker.image_exists()
        running  = self.docker.container_running()
        stopped  = self.docker.container_exists() and not running

        if running:
            ColorOutput.print("● Ollama is RUNNING", Colors.GREEN, bold=True)
        elif stopped:
            ColorOutput.print("● Ollama is STOPPED  →  press [3] to start", Colors.YELLOW, bold=True)
        elif image_ok:
            ColorOutput.print(
                "● Ollama image installed  →  press [2] to install a model, then [3] to start",
                Colors.YELLOW, bold=True,
            )
        else:
            ColorOutput.print(
                "● Ollama is NOT installed  →  start at step 1 below", Colors.GRAY, bold=True
            )

        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()

        # ── STEP 1 ───────────────────────────────────────────────────────────
        step1_label = (
            "✔ Ollama installed" if image_ok else "Ollama NOT installed  ← START HERE"
        )
        step1_color = Colors.GREEN if image_ok else Colors.YELLOW
        ColorOutput.print(f"  STEP 1 — INSTALL OLLAMA  ({step1_label})", step1_color, bold=True)
        print("    [1] Install Ollama  (download Docker image, ~1-2 GB)")
        print()

        # ── STEP 2 ───────────────────────────────────────────────────────────
        model_count = None
        if running:
            try:
                models = self.docker.list_models()
                model_count = max(0, len(models) - 1)
            except Exception:
                pass

        if model_count is not None and model_count > 0:
            step2_label = f"{model_count} model(s) installed"
            step2_color = Colors.GREEN
        elif not image_ok:
            step2_label = "complete step 1 first"
            step2_color = Colors.GRAY
        else:
            step2_label = "no models installed  ← DO THIS NEXT"
            step2_color = Colors.YELLOW

        ColorOutput.print(f"  STEP 2 — INSTALL A MODEL  ({step2_label})", step2_color, bold=True)
        print("    [2] Install Model         (pick from list or enter a name)")
        print("    [D] Uninstall Model")
        print("    [M] List Installed Models")
        print()

        # ── STEP 3 ───────────────────────────────────────────────────────────
        if running:
            step3_label, step3_color = "RUNNING",                    Colors.GREEN
        elif not image_ok:
            step3_label, step3_color = "complete steps 1 & 2 first", Colors.GRAY
        else:
            step3_label, step3_color = "ready to start",             Colors.CYAN

        ColorOutput.print(
            f"  STEP 3 — START / STOP OLLAMA  ({step3_label})", step3_color, bold=True
        )
        print("    [3] Start Ollama")
        print("    [4] Stop  Ollama")
        print("    [5] View Status")
        print()

        # ── USE ──────────────────────────────────────────────────────────────
        ColorOutput.print("  USE", Colors.CYAN, bold=True)
        print("    [6] Chat with Model")
        print("    [7] API Info & Network Access")
        print("    [L] Load Model into Memory")
        print("    [U] Unload Models from Memory")
        print()

        # ── TURBOQUANT ───────────────────────────────────────────────────────
        tq_running   = self.turboquant.is_server_running()
        tq_installed = self.turboquant.is_turboquant_installed()
        if tq_running:
            tq_status = (
                f"{Colors.GREEN}RUNNING{Colors.RESET}"
                f" — port {self.turboquant.config.get('port', 8000)}"
            )
        elif tq_installed:
            tq_status = f"{Colors.YELLOW}installed / stopped{Colors.RESET}"
        else:
            tq_status = f"{Colors.GRAY}not installed{Colors.RESET}"

        ColorOutput.print("  TURBOQUANT  (GPU Quantized Inference)", Colors.MAGENTA, bold=True)
        print(f"    [T] TurboQuant Server Manager  ({tq_status})")
        print()

        # ── ADVANCED ─────────────────────────────────────────────────────────
        ColorOutput.print("  ADVANCED", Colors.CYAN, bold=True)
        print("    [S] Settings  (GPU/CPU, Network, Port)")
        print("    [R] Recreate Container  (keeps models)")
        print("    [9] Remove Container & Image  (keeps models)")
        print("    [X] Full Uninstall  (deletes EVERYTHING including models)")
        print()

        print("    [0] Exit")
        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()

    # ── Application loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the application: check Docker, run first-time setup, then loop."""

        # ── Docker availability check ────────────────────────────────────────
        if not self.docker.is_docker_running():
            self._print_docker_not_running()
            return

        # ── Linux / WSL first-run requirements screen ────────────────────────
        if self.platform in (Platform.LINUX, Platform.WSL):
            self.setup.show_linux_requirements()

        # ── Main loop ────────────────────────────────────────────────────────
        while True:
            try:
                os.system("cls" if os.name == "nt" else "clear")
                self.show_main_menu()
                choice = input("Select an option: ").strip().upper()

                # Step 1
                if   choice == "1": self.menu.install_ollama()

                # Step 2 — model management
                elif choice == "2": self.menu.install_model()
                elif choice == "D": self.menu.uninstall_model()
                elif choice == "M": self.menu.list_models()

                # Step 3 — start / stop
                elif choice == "3":
                    self.docker.start_container()
                    input("\nPress Enter to continue...")
                elif choice == "4":
                    self.docker.stop_container()
                    input("\nPress Enter to continue...")
                elif choice == "5": self.menu.view_status()

                # Use
                elif choice == "6": self.menu.chat_menu()
                elif choice == "7": self.menu.show_api_info()
                elif choice == "L": self.menu.load_model_menu()
                elif choice == "U": self.menu.unload_models_menu()

                # TurboQuant
                elif choice == "T": self.turboquant.show_menu()

                # Advanced
                elif choice == "S": self.menu.handle_settings()
                elif choice == "R":
                    self.menu.recreate_container()
                    input("\nPress Enter to continue...")
                elif choice == "9":
                    self.docker.complete_removal()
                    input("\nPress Enter to continue...")
                elif choice == "X":
                    self.docker.full_uninstall()
                    input("\nPress Enter to continue...")

                elif choice == "0":
                    ColorOutput.info("Goodbye!")
                    break

            except KeyboardInterrupt:
                print("\n")
                ColorOutput.info("Goodbye!")
                break
            except Exception as exc:
                ColorOutput.error(f"An error occurred: {exc}")
                input("\nPress Enter to continue...")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _print_docker_not_running(self) -> None:
        print()
        ColorOutput.print("=" * 70, Colors.RED, bold=True)
        ColorOutput.print("  ❌ DOCKER IS NOT RUNNING", Colors.RED, bold=True)
        ColorOutput.print("=" * 70, Colors.RED, bold=True)
        print()
        ColorOutput.warning("The Docker daemon/engine is not currently running.")
        print()
        ColorOutput.print("To fix this:", Colors.CYAN, bold=True)
        print()

        plat = PlatformDetector.get_platform()
        if plat == Platform.WINDOWS:
            ColorOutput.print("  Windows:", Colors.YELLOW, bold=True)
            print("    1. Open Docker Desktop from the Start Menu")
            print("    2. Wait for it to fully start (system tray icon turns green)")
            print("    3. Try running this script again")
        elif plat == Platform.WSL:
            ColorOutput.print("  WSL:", Colors.YELLOW, bold=True)
            print("    1. Open Docker Desktop in Windows")
            print("    2. Ensure WSL2 integration is enabled in Docker Desktop settings")
            print("    3. Try running this script again")
        else:
            ColorOutput.print("  Linux:", Colors.YELLOW, bold=True)
            print("    1. sudo systemctl start docker")
            print("    2. Or to auto-start: sudo systemctl enable --now docker")
            print("    3. Try running this script again")
            print()
            ColorOutput.print("  If Docker is not installed:", Colors.GRAY)
            print("    sudo apt install docker.io   (Ubuntu/Debian)")
            print("    https://docs.docker.com/engine/install/")

        print()
        ColorOutput.print("=" * 70, Colors.RED, bold=True)
        print()
        ColorOutput.info("Please start Docker and try again.")
        print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        OllamaManager().run()
    except Exception as exc:
        print()
        ColorOutput.error(f"FATAL ERROR: {exc}")
        print()
        import traceback
        traceback.print_exc()
        print()
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
