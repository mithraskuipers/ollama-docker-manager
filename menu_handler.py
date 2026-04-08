"""
Ollama Docker Manager - Menu Handler
Contains all individual menu action methods called from the main OllamaManager loop.
Each method corresponds to one user-selectable option in the main menu.
"""

import json
import os
import re
import socket
import subprocess
from pathlib import Path

from utils import Platform, Colors, ColorOutput, PlatformDetector

# Optional: chat requires the requests library
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class MenuHandler:
    """
    Implements every menu action as a self-contained method.

    Receives references to the shared docker and config managers
    so it never needs to duplicate state.
    """

    def __init__(self, docker_manager, config_manager, platform: Platform):
        self.docker   = docker_manager
        self.cfg      = config_manager
        self.platform = platform

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Install Ollama
    # ══════════════════════════════════════════════════════════════════════════

    def install_ollama(self) -> None:
        """Download the Ollama Docker image."""
        ColorOutput.header("INSTALL OLLAMA")

        if self.docker.image_exists():
            ColorOutput.success("Ollama image is already installed!")
            ColorOutput.print(f"Image: {self.cfg.config.image_name}", Colors.CYAN)
            if not self.docker.container_exists():
                print()
                ColorOutput.info(
                    "Next step: press [2] to install a model, then [3] to start Ollama"
                )
        else:
            ColorOutput.info("This will download the Ollama Docker image")
            ColorOutput.print("  Estimated size: ~1-2 GB", Colors.GRAY)
            print()
            confirm = input("Continue with installation? (y/n): ").strip().lower()
            if confirm == "y":
                if self.docker.pull_image_with_progress():
                    print()
                    ColorOutput.success("Ollama installed successfully!")
                    ColorOutput.info(
                        "Next step: press [2] to install a model, then [3] to start Ollama"
                    )

        input("\nPress Enter to continue...")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Model management
    # ══════════════════════════════════════════════════════════════════════════

    def install_model(self) -> None:
        """Install (pull) a model from a list or by name."""
        ColorOutput.header("INSTALL MODEL")

        models_file = Path(self.cfg.config.models_list_file)
        model_name = ""

        if models_file.exists():
            try:
                with open(models_file, "r") as fh:
                    models_data = json.load(fh)

                model_list = models_data.get("models", [])

                if model_list:
                    import textwrap

                    ColorOutput.print("Available models from list:", Colors.CYAN, bold=True)
                    print()

                    for i, model_info in enumerate(model_list, 1):
                        name = model_info.get("name", "Unknown")
                        desc = model_info.get("description", "")
                        ColorOutput.print(f"[{i}] {name}", Colors.WHITE, bold=True)
                        if desc:
                            wrapped = textwrap.fill(
                                desc,
                                width=70,
                                initial_indent="    ",
                                subsequent_indent="    ",
                            )
                            ColorOutput.print(wrapped, Colors.GRAY)
                        print()

                    ColorOutput.print("─" * 60, Colors.GRAY)
                    print()
                    ColorOutput.print(
                        "Select a model by number, or type a custom model name:", Colors.WHITE
                    )
                    choice = input("Model: ").strip()

                    if choice.isdigit() and 1 <= int(choice) <= len(model_list):
                        model_name = model_list[int(choice) - 1].get("name", "")
                    else:
                        model_name = choice
                else:
                    ColorOutput.warning("Model list is empty")
                    model_name = input("Enter model name: ").strip()

            except json.JSONDecodeError as exc:
                ColorOutput.error(f"Error parsing models file: {exc}")
                model_name = input("Enter model name: ").strip()
            except Exception as exc:
                ColorOutput.error(f"Error reading models file: {exc}")
                model_name = input("Enter model name: ").strip()
        else:
            ColorOutput.print("Popular models:", Colors.CYAN)
            print("  • llama3.2    (3B — fast, good for chat)")
            print("  • llama3.2:1b (1B — very fast, basic tasks)")
            print("  • mistral     (7B — strong performance)")
            print("  • phi3        (3.8B — Microsoft, efficient)")
            print("  • qwen2.5:7b  (7B — multilingual)")
            print()
            model_name = input("Enter model name to install: ").strip()

        if model_name:
            self.docker.pull_model(model_name)

        input("\nPress Enter to continue...")

    def uninstall_model(self) -> None:
        """Remove an installed model."""
        ColorOutput.header("UNINSTALL MODEL")

        models = self.docker.list_models()
        if len(models) <= 1:
            ColorOutput.warning("No models installed")
            input("\nPress Enter to continue...")
            return

        models = models[1:]  # skip header line
        ColorOutput.print("Installed models:", Colors.CYAN, bold=True)
        for i, model in enumerate(models, 1):
            print(f"  [{i}] {model}")
        print()

        choice = input("Select model number to uninstall: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_name = models[int(choice) - 1].split()[0]
            self.docker.remove_model(model_name)

        input("\nPress Enter to continue...")

    def list_models(self) -> None:
        """List all installed models and their storage location."""
        ColorOutput.header("INSTALLED MODELS")

        models_dir = PlatformDetector.get_models_directory()
        ColorOutput.print(f"Storage location: {models_dir}", Colors.CYAN)
        print()

        models = self.docker.list_models()
        if models:
            for line in models:
                print(line)
        else:
            ColorOutput.warning("No models installed or container not running")

        input("\nPress Enter to continue...")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Container control
    # ══════════════════════════════════════════════════════════════════════════

    def view_status(self) -> None:
        """Show detailed container and storage status."""
        ColorOutput.header("OLLAMA STATUS")

        if not self.docker.is_docker_running():
            ColorOutput.error("Docker is not running!")
            input("\nPress Enter to continue...")
            return

        details = self.docker.get_container_details()
        if details:
            ColorOutput.print(f"Container Name: {details['name']}",    Colors.WHITE)
            ColorOutput.print(f"Image:          {details['image']}",   Colors.WHITE)
            ColorOutput.print(f"Created:        {details['created']}", Colors.WHITE)
            ColorOutput.print(f"Port:           {details['port']}",    Colors.WHITE)

            models_dir = self.cfg.config.models_dir
            ColorOutput.print(f"Models Dir:     {models_dir}", Colors.WHITE)

            plat = PlatformDetector.get_platform()
            if plat == Platform.WSL:
                print()
                ColorOutput.print("ℹ️  Platform: WSL (Windows Subsystem for Linux)", Colors.CYAN)
                ColorOutput.print("   Models stored in WSL filesystem, NOT Windows", Colors.GRAY)
            elif plat == Platform.WINDOWS:
                print()
                ColorOutput.print("ℹ️  Platform: Windows (native)", Colors.CYAN)
                ColorOutput.print("   Models stored in Windows filesystem", Colors.GRAY)
            elif plat == Platform.LINUX:
                print()
                ColorOutput.print("ℹ️  Platform: Linux (native)", Colors.CYAN)

            if Path(models_dir).exists():
                try:
                    total = sum(
                        f.stat().st_size
                        for f in Path(models_dir).rglob("*")
                        if f.is_file()
                    )
                    print()
                    ColorOutput.print("   Directory exists: ✓",                    Colors.GREEN)
                    ColorOutput.print(f"   Storage used: {total / 1024**3:.2f} GB", Colors.CYAN)
                except Exception:
                    print()
                    ColorOutput.print("   Directory exists: ✓", Colors.GREEN)
            else:
                print()
                ColorOutput.print(
                    "   Directory exists: ✗ (will be created when models are installed)",
                    Colors.YELLOW,
                )

            print()
            if details["gpu_enabled"]:
                ColorOutput.print("GPU: Enabled", Colors.GREEN)
            else:
                ColorOutput.print("GPU: Disabled (CPU only)", Colors.CYAN)

            print()
            if self.docker.container_running():
                ColorOutput.print("Status: Running", Colors.GREEN, bold=True)
            else:
                ColorOutput.print("Status: Stopped", Colors.YELLOW, bold=True)
        else:
            ColorOutput.warning("Container does not exist")

        input("\nPress Enter to continue...")

    def recreate_container(self) -> bool:
        """Stop, remove, and re-create the container (keeps image and models)."""
        ColorOutput.header("RECREATE CONTAINER")

        if not self.docker.container_exists():
            ColorOutput.warning("No container exists to recreate")
            return False

        ColorOutput.print("🔄 This will recreate the container with:", Colors.CYAN, bold=True)
        ColorOutput.print("  ✓ Latest DNS settings (fixes model download issues)", Colors.GREEN)
        ColorOutput.print("  ✓ Your current GPU/CPU and network settings",         Colors.GREEN)
        ColorOutput.print("  ✓ Docker image kept (no re-download)",                Colors.GREEN)
        ColorOutput.print("  ✓ ALL models preserved (stored outside container)",   Colors.GREEN, bold=True)

        models_dir = PlatformDetector.get_models_directory()
        if Path(models_dir).exists():
            print()
            ColorOutput.print("📁 Your models location:", Colors.CYAN, bold=True)
            ColorOutput.print(f"  {models_dir}", Colors.WHITE)
            blob_path = Path(models_dir) / "models" / "blobs"
            if blob_path.exists():
                try:
                    count = len(list(blob_path.glob("sha256-*")))
                    if count > 0:
                        ColorOutput.print(
                            f"  Contains {count} model file(s) — these will be preserved!",
                            Colors.GREEN,
                        )
                except Exception:
                    pass
        print()

        if self.docker.container_running():
            ColorOutput.info("Stopping running container...")
            if not self.docker.stop_container():
                ColorOutput.error("Failed to stop container")
                return False
            ColorOutput.success("Container stopped")

        ColorOutput.info("Removing old container instance...")
        if not self.docker.remove_container():
            ColorOutput.error("Failed to remove container")
            return False
        ColorOutput.success("Old container removed")

        ColorOutput.info("Creating new container with latest settings...")
        if self.docker.start_container():
            print()
            ColorOutput.success("✅ Container recreated successfully!")
            ColorOutput.print("  • New DNS settings active (8.8.8.8, 8.8.4.4, 1.1.1.1)", Colors.GREEN)
            ColorOutput.print("  • Model downloads should now work!",                      Colors.GREEN)
            ColorOutput.print(f"  • Your models are preserved in: {models_dir}",           Colors.CYAN)
            print()
            ColorOutput.print("💡 Try installing a model again now!", Colors.YELLOW)
            return True
        else:
            ColorOutput.error("Failed to create new container")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # USE — Chat and API info
    # ══════════════════════════════════════════════════════════════════════════

    def chat_menu(self) -> None:
        """Start an interactive chat session with an installed model."""
        ColorOutput.header("CHAT WITH MODEL")

        if not self.docker.container_running():
            ColorOutput.error("Ollama container is not running")
            ColorOutput.info("Please start the container first (option [3])")
            input("\nPress Enter to continue...")
            return

        models = self.docker.list_models()
        if len(models) <= 1:
            ColorOutput.warning("No models installed")
            input("\nPress Enter to continue...")
            return

        models = models[1:]
        ColorOutput.print("Installed models:", Colors.CYAN, bold=True)
        for i, model in enumerate(models, 1):
            print(f"  [{i}] {model}")
        print()

        choice = input("Select model number to chat with: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_name = models[int(choice) - 1].split()[0]
            self.docker.chat_with_model(model_name)
        else:
            ColorOutput.warning("Invalid selection")

        input("\nPress Enter to continue...")

    def show_api_info(self) -> None:
        """Display API endpoints and usage examples."""
        ColorOutput.header("API CONNECTION INFO")

        if not self.docker.container_running():
            ColorOutput.warning("Container is not running. Start the service first.")
            input("\nPress Enter to continue...")
            return

        port = self.cfg.config.ollama_port
        ColorOutput.success("Ollama API is running!")
        print()

        ColorOutput.print("LOCAL ACCESS (this computer only):", Colors.CYAN, bold=True)
        print(f"  http://localhost:{port}")
        print(f"  http://127.0.0.1:{port}")
        print()

        if not self.cfg.config.network_access:
            ColorOutput.print("NETWORK ACCESS: DISABLED ✗", Colors.GRAY, bold=True)
            print()
            ColorOutput.print("Only accessible from this computer (localhost only)", Colors.GRAY)
            print()
            ColorOutput.print("To enable network access:", Colors.CYAN)
            print("  Press [S] for Settings → [3] Enable Network Access")
            print()
            ColorOutput.print("─" * 60, Colors.GRAY)
            print()
            ColorOutput.print("EXAMPLE USAGE:", Colors.CYAN, bold=True)
            print()
            ColorOutput.print("Test connection:", Colors.WHITE)
            print(f"  curl http://localhost:{port}/api/tags")
            print()
            input("\nPress Enter to continue...")
            return

        # Network access is enabled
        ColorOutput.print("NETWORK ACCESS: ENABLED ✓", Colors.GREEN, bold=True)
        print()

        local_ip     = self._get_local_ip()
        tailscale_ip = self._get_tailscale_ip()

        if local_ip:
            ColorOutput.print("📡 LOCAL NETWORK ACCESS (same WiFi/LAN):", Colors.YELLOW, bold=True)
            print(f"  http://{local_ip}:{port}")
            print()
            ColorOutput.print("  Use this from:", Colors.CYAN)
            print("  • Other computers on the SAME network")
            print("  • Your phone when connected to the SAME WiFi")
            print()
            ColorOutput.print("  Test from another device:", Colors.GRAY)
            print(f"    curl http://{local_ip}:{port}/api/tags")
            print()

        if tailscale_ip:
            ColorOutput.print(
                "🔒 TAILSCALE VPN ACCESS (secure, from anywhere):", Colors.MAGENTA, bold=True
            )
            print(f"  http://{tailscale_ip}:{port}")
            print()
            ColorOutput.print("  Use this from:", Colors.CYAN)
            print("  • ANY device in your Tailscale network")
            print("  • Works even on different networks")
            print("  • Secure encrypted connection")
            print()
        else:
            ColorOutput.print("🔒 TAILSCALE VPN ACCESS:", Colors.GRAY, bold=True)
            ColorOutput.print("  Tailscale not detected", Colors.GRAY)
            print()
            ColorOutput.print("  To enable Tailscale:", Colors.CYAN)
            print("  1. Install: https://tailscale.com/download")
            print("  2. Run: tailscale up")
            print()

        if local_ip and not tailscale_ip:
            ColorOutput.print("⚠️  PUBLIC WIFI WARNING:", Colors.RED, bold=True)
            print("  On public WiFi, local network access usually won't work.")
            print("  Use Tailscale instead for secure remote access.")
            print()

        ColorOutput.print("─" * 60, Colors.GRAY)
        print()

        # Connection summary
        ColorOutput.print("CONNECTION SUMMARY:", Colors.CYAN, bold=True)
        print(f"  ✓ Localhost:     http://localhost:{port}")
        if local_ip:
            print(f"  ✓ Local Network: http://{local_ip}:{port}")
        if tailscale_ip:
            print(f"  ✓ Tailscale VPN: http://{tailscale_ip}:{port}")

        input("\nPress Enter to continue...")

    # ══════════════════════════════════════════════════════════════════════════
    # Memory management
    # ══════════════════════════════════════════════════════════════════════════

    def load_model_menu(self) -> None:
        """Pre-load a model into GPU/CPU memory."""
        ColorOutput.header("LOAD MODEL INTO MEMORY")

        if not self.docker.container_running():
            ColorOutput.error("Ollama container is not running")
            input("\nPress Enter to continue...")
            return

        models = self.docker.list_models()
        if len(models) <= 1:
            ColorOutput.warning("No models installed")
            input("\nPress Enter to continue...")
            return

        models = models[1:]
        ColorOutput.print("Installed models:", Colors.CYAN, bold=True)
        for i, model in enumerate(models, 1):
            print(f"  [{i}] {model}")
        print()

        loaded = self.docker.get_loaded_models()
        if loaded:
            ColorOutput.print("Currently loaded in memory:", Colors.YELLOW, bold=True)
            for m in loaded:
                print(f"  • {m['name']}")
            print()

        choice = input("Select model number to load into memory: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_name = models[int(choice) - 1].split()[0]
            self.docker.load_model(model_name)
        else:
            ColorOutput.error("Invalid selection")

        input("\nPress Enter to continue...")

    def unload_models_menu(self) -> None:
        """Unload one or all models from memory."""
        ColorOutput.header("UNLOAD MODELS FROM MEMORY")

        if not self.docker.container_running():
            ColorOutput.error("Ollama container is not running")
            input("\nPress Enter to continue...")
            return

        loaded = self.docker.get_loaded_models()
        if not loaded:
            ColorOutput.info("No models are currently loaded in memory")
            input("\nPress Enter to continue...")
            return

        ColorOutput.print("Currently loaded models:", Colors.CYAN, bold=True)
        for i, m in enumerate(loaded, 1):
            print(f"  [{i}] {m['full_line']}")
        print()
        print("  [A] Unload all models (graceful)")
        print("  [F] Force unload all (restart container)")
        print("  [0] Cancel")
        print()

        choice = input("Select model number to unload (or A/F for all): ").strip().upper()

        if choice == "0":
            ColorOutput.info("Operation cancelled")

        elif choice == "F":
            ColorOutput.warning("This will restart the Ollama container to clear all models")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == "y":
                ColorOutput.info("Restarting container...")
                if self.docker.stop_container() and self.docker.start_container():
                    print()
                    ColorOutput.success("Container restarted — all models cleared from memory")
                else:
                    ColorOutput.error("Failed to restart container")
            else:
                ColorOutput.info("Operation cancelled")

        elif choice == "A":
            ColorOutput.info("Unloading all models...")
            ok_count = 0
            failed = []
            for m in loaded:
                if self.docker.unload_model(m["name"], force=False):
                    ok_count += 1
                else:
                    failed.append(m["name"])
            print()
            ColorOutput.success(f"Unloaded {ok_count} of {len(loaded)} models")
            if failed:
                print()
                ColorOutput.warning("Failed to unload:")
                for name in failed:
                    print(f"  • {name}")
                print()
                ColorOutput.print(
                    "Try force unload (option F) to restart container and clear all", Colors.YELLOW
                )

        elif choice.isdigit() and 1 <= int(choice) <= len(loaded):
            m = loaded[int(choice) - 1]
            if not self.docker.unload_model(m["name"], force=False):
                print()
                ColorOutput.warning("Standard unload failed")
                retry = input("Try force unload? (y/n): ").strip().lower()
                if retry == "y":
                    self.docker.unload_model(m["name"], force=True)
        else:
            ColorOutput.error("Invalid selection")

        input("\nPress Enter to continue...")

    # ══════════════════════════════════════════════════════════════════════════
    # Settings
    # ══════════════════════════════════════════════════════════════════════════

    def handle_settings(self) -> None:
        """Interactive settings menu (GPU, network, port, concurrent models)."""
        ColorOutput.header("SETTINGS")

        cfg = self.cfg.config
        max_models = cfg.max_concurrent_models

        # ── Current values ──────────────────────────────────────────────────
        ColorOutput.print("GPU ACCELERATION:", Colors.CYAN, bold=True)
        if cfg.use_gpu:
            ColorOutput.print("  Status: ENABLED",          Colors.GREEN)
        else:
            ColorOutput.print("  Status: DISABLED (CPU only)", Colors.GRAY)
        print()

        ColorOutput.print("NETWORK ACCESS:", Colors.CYAN, bold=True)
        if cfg.network_access:
            ColorOutput.print("  Status: ENABLED (accessible from other computers)", Colors.YELLOW)
        else:
            ColorOutput.print("  Status: DISABLED (localhost only)", Colors.GRAY)
        print()

        ColorOutput.print("PORT:", Colors.CYAN, bold=True)
        ColorOutput.print(f"  Current port: {cfg.ollama_port}", Colors.WHITE)
        print()

        ColorOutput.print("CONCURRENT MODEL LIMIT:", Colors.CYAN, bold=True)
        if max_models == 1:
            ColorOutput.print(f"  Limit: {max_models} model (auto-unload enabled)", Colors.GREEN)
            ColorOutput.print(
                "  When loading a new model, the previous one will be automatically unloaded",
                Colors.GRAY,
            )
        elif max_models == 0:
            ColorOutput.print("  Limit: Unlimited (all models stay loaded)", Colors.YELLOW)
        else:
            ColorOutput.print(f"  Limit: {max_models} models at once", Colors.GREEN)
            ColorOutput.print(
                f"  When loading model #{max_models + 1}, the oldest will be unloaded", Colors.GRAY
            )
        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()

        # ── GPU diagnostics ──────────────────────────────────────────────────
        ColorOutput.print("Checking GPU availability...", Colors.GRAY)
        gpu_info = PlatformDetector.get_gpu_diagnostics()
        print()

        if gpu_info["available"]:
            ColorOutput.success("GPU acceleration is available on this system!")
            if "gpu_name" in gpu_info:
                ColorOutput.print(f"  Detected GPU: {gpu_info['gpu_name']}", Colors.WHITE)
        else:
            ColorOutput.warning("GPU acceleration is not available")
            if gpu_info.get("issues"):
                print()
                ColorOutput.print("Issues detected:", Colors.YELLOW, bold=True)
                for issue in gpu_info["issues"]:
                    print(f"  • {issue}")
            if gpu_info.get("recommendations"):
                print()
                ColorOutput.print("Recommendations:", Colors.CYAN, bold=True)
                for rec in gpu_info["recommendations"]:
                    print(f"  {rec}")

        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()

        # ── Options ──────────────────────────────────────────────────────────
        ColorOutput.print("SETTINGS OPTIONS:", Colors.CYAN, bold=True)
        print()
        ColorOutput.print("GPU Settings:", Colors.WHITE)
        print("  [1] Enable GPU Acceleration")
        print("  [2] Disable GPU (CPU only)")
        print()
        ColorOutput.print("Network Settings:", Colors.WHITE)
        print("  [3] Enable Network Access (allow other computers)")
        print("  [4] Disable Network Access (localhost only)")
        print()
        ColorOutput.print("Port Settings:", Colors.WHITE)
        print(f"  [5] Change Port (current: {cfg.ollama_port})")
        print()
        ColorOutput.print("Model Memory Management:", Colors.WHITE)
        print(f"  [6] Set Max Concurrent Models (current: {max_models if max_models > 0 else 'unlimited'})")
        print()
        print("  [0] Back to main menu")
        print()

        choice = input("Select option: ").strip()

        if choice == "1":
            self._settings_gpu(enable=True, gpu_info=gpu_info)
        elif choice == "2":
            self._settings_gpu(enable=False, gpu_info=gpu_info)
        elif choice == "3":
            self._settings_network(enable=True)
        elif choice == "4":
            self._settings_network(enable=False)
        elif choice == "5":
            self._settings_port()
        elif choice == "6":
            self._settings_concurrent_models()

        input("\nPress Enter to continue...")

    # ── Settings sub-handlers ──────────────────────────────────────────────────

    def _settings_gpu(self, enable: bool, gpu_info: dict) -> None:
        if enable and not gpu_info["available"]:
            print()
            ColorOutput.warning("GPU is not available on this system")
            ColorOutput.print(
                "  Enabling GPU mode anyway (will fail if requirements not met)", Colors.GRAY
            )
            print()
            if input("Continue? (y/n): ").strip().lower() != "y":
                return

        self.cfg.config.use_gpu = enable
        self.cfg.save_config()
        print()
        ColorOutput.success("GPU mode enabled!" if enable else "CPU mode enabled!")
        self._offer_recreate()

    def _settings_network(self, enable: bool) -> None:
        if enable:
            print()
            ColorOutput.warning("⚠ SECURITY WARNING ⚠")
            print()
            print("Enabling network access will make Ollama accessible from ANY computer")
            print("on your network. There is NO authentication or security built in.")
            print()
            ColorOutput.print("Only enable if you trust everyone on your network.", Colors.CYAN)
            print()
            if input("Enable network access? (yes/no): ").strip().lower() != "yes":
                ColorOutput.info("Network access not enabled")
                return

        if not enable:
            # Offer to clean up any existing firewall rule
            firewall = self.docker._check_firewall_status()
            if firewall["active"] and firewall["rule_exists"]:
                print()
                ColorOutput.print("Firewall cleanup:", Colors.CYAN, bold=True)
                ColorOutput.print("  A firewall rule exists for network access", Colors.GRAY)
                print()
                if input("Remove firewall rule? (y/n): ").strip().lower() == "y":
                    if self.docker._check_sudo_access():
                        ok, msg = self.docker._remove_firewall_rule()
                        print()
                        (ColorOutput.success if ok else ColorOutput.warning)(f"  {msg}")
                    else:
                        print()
                        ColorOutput.print("  Manual cleanup required (no sudo access):", Colors.YELLOW)
                        port = self.cfg.config.ollama_port
                        if firewall["type"] == "ufw":
                            print(f"    sudo ufw delete allow {port}/tcp")
                        elif firewall["type"] == "iptables":
                            print(f"    sudo iptables -D INPUT -p tcp --dport {port} -j ACCEPT")
                else:
                    ColorOutput.print("  Firewall rule kept", Colors.GRAY)

        self.cfg.config.network_access = enable
        self.cfg.save_config()
        print()
        ColorOutput.success(
            "Network access enabled!" if enable else "Network access disabled! (localhost only)"
        )
        self._offer_recreate()

    def _settings_port(self) -> None:
        print()
        ColorOutput.print("CHANGE PORT", Colors.CYAN, bold=True)
        print()
        ColorOutput.print(f"Current port: {self.cfg.config.ollama_port}", Colors.WHITE)
        print()
        ColorOutput.print("Common ports:", Colors.GRAY)
        print("  • 11434 (default)")
        print("  • 11435, 11436, etc. (if 11434 is in use)")
        print("  • 8080, 8081, 8082 (alternative)")
        print()

        new_port = input("Enter new port number (or press Enter to cancel): ").strip()
        if not new_port:
            ColorOutput.info("Port change cancelled")
            return
        if not new_port.isdigit():
            ColorOutput.error("Invalid port number")
            return

        port_num = int(new_port)
        if not (1024 <= port_num <= 65535):
            ColorOutput.error("Port must be between 1024 and 65535")
            return

        old_port = self.cfg.config.ollama_port
        self.cfg.config.ollama_port = port_num
        self.cfg.save_config()
        print()
        ColorOutput.success(f"Port changed from {old_port} to {port_num}!")
        self._offer_recreate()

    def _settings_concurrent_models(self) -> None:
        print()
        ColorOutput.print("SET MAX CONCURRENT MODELS", Colors.CYAN, bold=True)
        print()
        current = self.cfg.config.max_concurrent_models
        ColorOutput.print(
            f"Current limit: {current if current > 0 else 'unlimited'}", Colors.WHITE
        )
        print()
        ColorOutput.print("What this does:", Colors.YELLOW, bold=True)
        print("  • Limits how many models can be loaded in memory at once")
        print("  • When you load a new model and reach the limit, the oldest is auto-unloaded")
        print()
        ColorOutput.print("Recommended values:", Colors.GRAY)
        print("  • 1 = One model at a time (good for low memory)")
        print("  • 2-3 = Multiple models (if you have enough RAM/VRAM)")
        print("  • 0 = Unlimited")
        print()

        new_limit = input("Enter max concurrent models (0 for unlimited): ").strip()
        if not new_limit:
            ColorOutput.info("Setting change cancelled")
            return
        if not new_limit.isdigit():
            ColorOutput.error("Invalid number")
            return

        limit_num = int(new_limit)
        if not (0 <= limit_num <= 20):
            ColorOutput.error("Limit must be between 0 and 20")
            return

        self.cfg.config.max_concurrent_models = limit_num
        self.cfg.save_config()
        print()
        if limit_num == 0:
            ColorOutput.success(f"Concurrent model limit set to unlimited!")
        else:
            ColorOutput.success(f"Concurrent model limit set to {limit_num}!")
        ColorOutput.info("This setting takes effect immediately for new model loads")

    def _offer_recreate(self) -> None:
        """If a container exists, offer to recreate it so settings take effect."""
        if not self.docker.container_exists():
            return
        print()
        ColorOutput.warning("Container needs to be recreated for this to take effect")
        print()
        if input("Recreate container now? (y/n): ").strip().lower() == "y":
            self.recreate_container()
        else:
            print()
            ColorOutput.info("Container will keep old settings until recreated")
            ColorOutput.print("  Use [R] Recreate Container from the main menu.", Colors.GRAY)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return ""

    @staticmethod
    def _get_tailscale_ip() -> str:
        try:
            result = subprocess.run(
                ["ip", "addr", "show", "tailscale0"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return ""
