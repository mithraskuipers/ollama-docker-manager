#!/usr/bin/env python3
"""
Ollama Docker Manager - Cross-Platform Edition
Main application file - manages UI, menus, and orchestrates Docker and Config operations

Supports both Windows and Linux (WSL)

Dependencies:
- Docker
- Python 3.7+
- requests (install with: pip install requests)
"""

import os
import sys
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

# Import from our modules
from utils import (
    Platform, Colors, OllamaConfig, PlatformDetector, ColorOutput, VenvManager
)
from docker_manager import DockerManager
from turboquant_manager import TurboQuantManager

# For chat functionality
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Platform-specific imports for chat interruption
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty
    import select


class ConfigManager:
    """Manage configuration file"""
    
    def __init__(self):
        self.config = OllamaConfig()
        self.load_config()
        # Set models directory based on platform
        self.config.models_dir = PlatformDetector.get_models_directory()
        # Set container name based on platform
        self.config.container_name = self._get_platform_container_name()
    
    def _get_platform_container_name(self) -> str:
        """Get platform-specific container name"""
        plat = PlatformDetector.get_platform()
        
        if plat == Platform.WINDOWS:
            return "ollama-win"
        elif plat == Platform.LINUX:
            return "ollama-linux"
        elif plat == Platform.WSL:
            return "ollama-wsl"
        else:
            return "ollama"  # Fallback
    
    def load_config(self):
        """Load configuration from file"""
        if Path(self.config.config_file).exists():
            try:
                with open(self.config.config_file, 'r') as f:
                    data = json.load(f)
                    self.config.use_gpu = data.get('UseGPU', False)
                    self.config.network_access = data.get('NetworkAccess', False)
                    self.config.ollama_port = data.get('OllamaPort', 11434)
                    self.config.max_concurrent_models = data.get('MaxConcurrentModels', 1)
                    # Load container_name if present, otherwise will use platform default
                    saved_container = data.get('ContainerName')
                    if saved_container:
                        self.config.container_name = saved_container
            except Exception as e:
                ColorOutput.warning(f"Could not load config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config.config_file, 'w') as f:
                json.dump({
                    'UseGPU': self.config.use_gpu,
                    'NetworkAccess': self.config.network_access,
                    'OllamaPort': self.config.ollama_port,
                    'ContainerName': self.config.container_name,
                    'MaxConcurrentModels': self.config.max_concurrent_models
                }, f, indent=4)
            return True
        except Exception as e:
            ColorOutput.error(f"Could not save config: {e}")
            return False


class OllamaManager:
    """Main application class"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.docker = DockerManager(self.config_manager.config)
        self.platform = PlatformDetector.get_platform()
        self.turboquant = TurboQuantManager()
    
    def show_main_menu(self):
        """Display main menu"""
        # Banner
        ColorOutput.print("=" * 60, Colors.CYAN, bold=True)
        ColorOutput.print("       🦙 OLLAMA DOCKER MANAGER - Cross Platform", Colors.CYAN, bold=True)
        ColorOutput.print("=" * 60, Colors.CYAN, bold=True)
        print()

        # Platform info
        ColorOutput.print(f"Platform: {self.platform.value}", Colors.GRAY)
        ColorOutput.print(f"Container: {self.config_manager.config.container_name}  |  Port: {self.config_manager.config.ollama_port}", Colors.GRAY)
        print()

        # ── Status block with next-step hint ──────────────────────────────────
        image_ok    = self.docker.image_exists()
        running     = self.docker.container_running()
        stopped     = self.docker.container_exists() and not running

        if running:
            ColorOutput.print("● Ollama is RUNNING", Colors.GREEN, bold=True)
        elif stopped:
            ColorOutput.print("● Ollama is STOPPED  →  press [3] to start", Colors.YELLOW, bold=True)
        elif image_ok:
            ColorOutput.print("● Ollama image installed  →  press [2] to install a model, then [3] to start", Colors.YELLOW, bold=True)
        else:
            ColorOutput.print("● Ollama is NOT installed  →  start at step 1 below", Colors.GRAY, bold=True)

        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()

        # ── STEP 1 ────────────────────────────────────────────────────────────
        if image_ok:
            step1_label = "✔ Ollama installed"
            step1_color = Colors.GREEN
        else:
            step1_label = "Ollama NOT installed  ← START HERE"
            step1_color = Colors.YELLOW

        ColorOutput.print(f"  STEP 1 — INSTALL OLLAMA  ({step1_label})", step1_color, bold=True)
        print("    [1] Install Ollama  (download Docker image, ~1-2 GB)")
        print()

        # ── STEP 2 ────────────────────────────────────────────────────────────
        # Try to count installed models (needs container running; show hint otherwise)
        model_count = None
        if running:
            try:
                models = self.docker.list_models()
                model_count = max(0, len(models) - 1)   # subtract header line
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

        # ── STEP 3 ────────────────────────────────────────────────────────────
        if running:
            step3_label = "RUNNING"
            step3_color = Colors.GREEN
        elif not image_ok:
            step3_label = "complete steps 1 & 2 first"
            step3_color = Colors.GRAY
        else:
            step3_label = "ready to start"
            step3_color = Colors.CYAN

        ColorOutput.print(f"  STEP 3 — START / STOP OLLAMA  ({step3_label})", step3_color, bold=True)
        print("    [3] Start Ollama")
        print("    [4] Stop  Ollama")
        print("    [5] View Status")
        print()

        # ── USE ───────────────────────────────────────────────────────────────
        ColorOutput.print("  USE", Colors.CYAN, bold=True)
        print("    [6] Chat with Model")
        print("    [7] API Info & Network Access")
        print("    [L] Load Model into Memory")
        print("    [U] Unload Models from Memory")
        print()

        # ── TURBOQUANT ────────────────────────────────────────────────────────
        tq_running   = self.turboquant.is_server_running()
        tq_installed = self.turboquant.is_turboquant_installed()
        if tq_running:
            tq_status = f"{Colors.GREEN}RUNNING{Colors.RESET} — port {self.turboquant.config.get('port', 8000)}"
        elif tq_installed:
            tq_status = f"{Colors.YELLOW}installed / stopped{Colors.RESET}"
        else:
            tq_status = f"{Colors.GRAY}not installed{Colors.RESET}"
        ColorOutput.print("  TURBOQUANT  (GPU Quantized Inference)", Colors.MAGENTA, bold=True)
        print(f"    [T] TurboQuant Server Manager  ({tq_status})")
        print()

        # ── ADVANCED ──────────────────────────────────────────────────────────
        ColorOutput.print("  ADVANCED", Colors.CYAN, bold=True)
        print("    [S] Settings (GPU/CPU, Network, Port)")
        print("    [R] Recreate Container  (keeps models)")
        print("    [9] Remove Container & Image  (keeps models)")
        print("    [X] Full Uninstall  (deletes EVERYTHING including models)")
        print()

        print("    [0] Exit")
        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()
    
    def recreate_container(self):
        """Recreate the container with current settings (keeps image and models)"""
        ColorOutput.header("RECREATE CONTAINER")
        
        if not self.docker.container_exists():
            ColorOutput.warning("No container exists to recreate")
            return False
        
        ColorOutput.print("🔄 This will recreate the container with:", Colors.CYAN, bold=True)
        ColorOutput.print("  ✓ Latest DNS settings (fixes model download issues)", Colors.GREEN)
        ColorOutput.print("  ✓ Your current GPU/CPU and network settings", Colors.GREEN)
        ColorOutput.print("  ✓ Docker image kept (no re-download)", Colors.GREEN)
        ColorOutput.print("  ✓ ALL models preserved (stored outside container)", Colors.GREEN, bold=True)
        
        # Verify models directory exists and show its location
        models_dir = PlatformDetector.get_models_directory()
        if Path(models_dir).exists():
            print()
            ColorOutput.print(f"📁 Your models location:", Colors.CYAN, bold=True)
            ColorOutput.print(f"  {models_dir}", Colors.WHITE)
            
            # Show model files if they exist
            models_path = Path(models_dir)
            blob_path = models_path / "models" / "blobs"
            if blob_path.exists():
                try:
                    blob_count = len(list(blob_path.glob("sha256-*")))
                    if blob_count > 0:
                        ColorOutput.print(f"  Contains {blob_count} model file(s) - these will be preserved!", Colors.GREEN)
                except:
                    pass
        print()
        
        # Stop if running
        if self.docker.container_running():
            ColorOutput.info("Stopping running container...")
            if not self.docker.stop_container():
                ColorOutput.error("Failed to stop container")
                return False
            ColorOutput.success("Container stopped")
        
        # Remove container
        ColorOutput.info("Removing old container instance...")
        if self.docker.remove_container():
            ColorOutput.success("Old container removed")
        else:
            ColorOutput.error("Failed to remove container")
            return False
        
        # Create new container
        ColorOutput.info("Creating new container with latest settings...")
        if self.docker.start_container():
            print()
            ColorOutput.success("✅ Container recreated successfully!")
            ColorOutput.print("  • New DNS settings active (8.8.8.8, 8.8.4.4, 1.1.1.1)", Colors.GREEN)
            ColorOutput.print("  • Model downloads should now work!", Colors.GREEN)
            ColorOutput.print(f"  • Your models are preserved in: {models_dir}", Colors.CYAN)
            return True
        else:
            ColorOutput.error("Failed to create new container")
            return False
    
    def handle_settings(self):
        """Handle settings menu"""
        ColorOutput.header("SETTINGS")
        
        # Current GPU setting
        ColorOutput.print("GPU ACCELERATION:", Colors.CYAN, bold=True)
        if self.config_manager.config.use_gpu:
            ColorOutput.print("  Status: ENABLED", Colors.GREEN)
        else:
            ColorOutput.print("  Status: DISABLED (CPU only)", Colors.GRAY)
        
        print()
        
        # Current Network Access setting
        ColorOutput.print("NETWORK ACCESS:", Colors.CYAN, bold=True)
        if self.config_manager.config.network_access:
            ColorOutput.print("  Status: ENABLED (accessible from other computers)", Colors.YELLOW)
        else:
            ColorOutput.print("  Status: DISABLED (localhost only)", Colors.GRAY)
        
        print()
        
        # Current Port setting
        ColorOutput.print("PORT:", Colors.CYAN, bold=True)
        ColorOutput.print(f"  Current port: {self.config_manager.config.ollama_port}", Colors.WHITE)
        
        print()
        
        # Current Max Concurrent Models setting
        ColorOutput.print("CONCURRENT MODEL LIMIT:", Colors.CYAN, bold=True)
        max_models = self.config_manager.config.max_concurrent_models
        if max_models == 1:
            ColorOutput.print(f"  Limit: {max_models} model (auto-unload enabled)", Colors.GREEN)
            ColorOutput.print("  When loading a new model, the previous one will be automatically unloaded", Colors.GRAY)
        elif max_models == 0:
            ColorOutput.print("  Limit: Unlimited (all models stay loaded)", Colors.YELLOW)
        else:
            ColorOutput.print(f"  Limit: {max_models} models at once", Colors.GREEN)
            ColorOutput.print(f"  When loading model #{max_models + 1}, the oldest will be unloaded", Colors.GRAY)
        
        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()
        
        # Check GPU availability
        ColorOutput.print("Checking GPU availability...", Colors.GRAY)
        gpu_info = PlatformDetector.get_gpu_diagnostics()
        
        print()
        
        if gpu_info['available']:
            ColorOutput.success("GPU acceleration is available on this system!")
            if 'gpu_name' in gpu_info:
                ColorOutput.print(f"  Detected GPU: {gpu_info['gpu_name']}", Colors.WHITE)
        else:
            ColorOutput.warning("GPU acceleration is not available")
            if gpu_info['issues']:
                print()
                ColorOutput.print("Issues detected:", Colors.YELLOW, bold=True)
                for issue in gpu_info['issues']:
                    print(f"  • {issue}")
            
            if gpu_info['recommendations']:
                print()
                ColorOutput.print("Recommendations:", Colors.CYAN, bold=True)
                for rec in gpu_info['recommendations']:
                    print(f"  {rec}")
        
        print()
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()
        
        # Options
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
        print(f"  [5] Change Port (current: {self.config_manager.config.ollama_port})")
        print()
        ColorOutput.print("Model Memory Management:", Colors.WHITE)
        print(f"  [6] Set Max Concurrent Models (current: {max_models if max_models > 0 else 'unlimited'})")
        print()
        print("  [0] Back to main menu")
        print()
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            if not gpu_info['available']:
                print()
                ColorOutput.warning("GPU is not available on this system")
                ColorOutput.print("  Enabling GPU mode anyway (will fail if requirements not met)", Colors.GRAY)
                print()
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    return
            
            self.config_manager.config.use_gpu = True
            self.config_manager.save_config()
            print()
            ColorOutput.success("GPU mode enabled!")
            
            if self.docker.container_exists():
                print()
                ColorOutput.warning("Container needs to be recreated for this to take effect")
                print()
                choice = input("Recreate container now? (y/n): ").strip().lower()
                if choice == 'y':
                    self.recreate_container()
                    # Don't add input() here - handle_settings has final prompt
                else:
                    print()
                    ColorOutput.info("Container will keep old settings until recreated")
                    ColorOutput.print("  To recreate later:", Colors.CYAN)
                    ColorOutput.print("    [5] Stop → [4] Start", Colors.GRAY)
                    ColorOutput.print("  Or run Settings again and choose 'y'", Colors.GRAY)
        
        elif choice == '2':
            self.config_manager.config.use_gpu = False
            self.config_manager.save_config()
            print()
            ColorOutput.success("CPU mode enabled!")
            
            if self.docker.container_exists():
                print()
                ColorOutput.warning("Container needs to be recreated for this to take effect")
                print()
                choice = input("Recreate container now? (y/n): ").strip().lower()
                if choice == 'y':
                    self.recreate_container()
                    # Don't add input() here - handle_settings has final prompt
                else:
                    print()
                    ColorOutput.info("Container will keep old settings until recreated")
                    ColorOutput.print("  To recreate later:", Colors.CYAN)
                    ColorOutput.print("    [5] Stop → [4] Start", Colors.GRAY)
                    ColorOutput.print("  Or run Settings again and choose 'y'", Colors.GRAY)
        
        elif choice == '3':
            print()
            ColorOutput.warning("⚠ SECURITY WARNING ⚠")
            print()
            print("Enabling network access will make Ollama accessible from ANY computer")
            print("on your network. This means:")
            print()
            print("  • Other computers can use your GPU/CPU resources")
            print("  • Anyone on your network can access all your models")
            print("  • There is NO authentication or security")
            print()
            ColorOutput.print("Only enable this if:", Colors.CYAN)
            print("  ✓ You trust everyone on your network")
            print("  ✓ You're on a private home/office network")
            print("  ✓ You want to share Ollama with other devices you own")
            print()
            ColorOutput.print("DO NOT enable if:", Colors.RED)
            print("  ✗ You're on public WiFi")
            print("  ✗ You're on a shared/untrusted network")
            print()
            
            confirm = input("Enable network access? (yes/no): ").strip().lower()
            if confirm == 'yes':
                self.config_manager.config.network_access = True
                self.config_manager.save_config()
                print()
                ColorOutput.success("Network access enabled!")
                
                if self.docker.container_exists():
                    print()
                    ColorOutput.warning("Container needs to be recreated for this to take effect")
                    print()
                    choice = input("Recreate container now? (y/n): ").strip().lower()
                    if choice == 'y':
                        if self.recreate_container():
                            print()
                            ColorOutput.info("Use option [8] to see network connection info")
                        # Don't add input() here - handle_settings has final prompt
                    else:
                        print()
                        ColorOutput.info("Container will keep old settings until recreated")
                        ColorOutput.print("  To recreate later:", Colors.CYAN)
                        ColorOutput.print("    [5] Stop → [4] Start", Colors.GRAY)
                        ColorOutput.print("  Or run Settings again and choose 'y'", Colors.GRAY)
                        print()
                        ColorOutput.info("After recreating, use option [8] to see network connection info")
            else:
                ColorOutput.info("Network access not enabled")
        
        elif choice == '4':
            print()
            ColorOutput.print("Disabling network access...", Colors.CYAN)
            print()
            
            # Check if firewall rule exists
            firewall_status = self.docker._check_firewall_status()
            should_cleanup_firewall = (firewall_status['active'] and firewall_status['rule_exists'])
            
            self.config_manager.config.network_access = False
            self.config_manager.save_config()
            print()
            ColorOutput.success("Network access disabled! (localhost only)")
            
            # Offer to clean up firewall rule
            if should_cleanup_firewall:
                print()
                ColorOutput.print("Firewall cleanup:", Colors.CYAN, bold=True)
                ColorOutput.print("  A firewall rule exists for network access", Colors.GRAY)
                print()
                cleanup = input("Remove firewall rule? (y/n): ").strip().lower()
                
                if cleanup == 'y':
                    if self.docker._check_sudo_access():
                        success, message = self.docker._remove_firewall_rule()
                        print()
                        if success:
                            ColorOutput.success(f"  {message}")
                        else:
                            ColorOutput.warning(f"  {message}")
                            print()
                            ColorOutput.print("  Manual cleanup command:", Colors.CYAN)
                            if firewall_status['type'] == 'ufw':
                                print(f"    sudo ufw delete allow {self.config_manager.config.ollama_port}/tcp")
                            elif firewall_status['type'] == 'iptables':
                                print(f"    sudo iptables -D INPUT -p tcp --dport {self.config_manager.config.ollama_port} -j ACCEPT")
                    else:
                        print()
                        ColorOutput.print("  Manual cleanup required (no sudo access):", Colors.YELLOW)
                        if firewall_status['type'] == 'ufw':
                            print(f"    sudo ufw delete allow {self.config_manager.config.ollama_port}/tcp")
                        elif firewall_status['type'] == 'iptables':
                            print(f"    sudo iptables -D INPUT -p tcp --dport {self.config_manager.config.ollama_port} -j ACCEPT")
                else:
                    ColorOutput.print("  Firewall rule kept (you can remove it manually later)", Colors.GRAY)
            
            if self.docker.container_exists():
                print()
                ColorOutput.warning("Container needs to be recreated for this to take effect")
                print()
                choice = input("Recreate container now? (y/n): ").strip().lower()
                if choice == 'y':
                    self.recreate_container()
                    # Don't add input() here - handle_settings has final prompt
                else:
                    print()
                    ColorOutput.info("Container will keep old settings until recreated")
                    ColorOutput.print("  To recreate later:", Colors.CYAN)
                    ColorOutput.print("    [5] Stop → [4] Start", Colors.GRAY)
                    ColorOutput.print("  Or run Settings again and choose 'y'", Colors.GRAY)
        
        elif choice == '5':
            print()
            ColorOutput.print("CHANGE PORT", Colors.CYAN, bold=True)
            print()
            ColorOutput.print(f"Current port: {self.config_manager.config.ollama_port}", Colors.WHITE)
            print()
            ColorOutput.print("Common ports:", Colors.GRAY)
            print("  • 11434 (default)")
            print("  • 11435, 11436, etc. (if 11434 is in use)")
            print("  • 8080, 8081, 8082 (alternative)")
            print()
            
            try:
                new_port = input("Enter new port number (or press Enter to cancel): ").strip()
                if not new_port:
                    ColorOutput.info("Port change cancelled")
                elif not new_port.isdigit():
                    ColorOutput.error("Invalid port number")
                else:
                    port_num = int(new_port)
                    if port_num < 1024 or port_num > 65535:
                        ColorOutput.error("Port must be between 1024 and 65535")
                    else:
                        old_port = self.config_manager.config.ollama_port
                        self.config_manager.config.ollama_port = port_num
                        self.config_manager.save_config()
                        print()
                        ColorOutput.success(f"Port changed from {old_port} to {port_num}!")
                        
                        if self.docker.container_exists():
                            print()
                            ColorOutput.warning("Container needs to be recreated for this to take effect")
                            print()
                            choice = input("Recreate container now? (y/n): ").strip().lower()
                            if choice == 'y':
                                self.recreate_container()
                            else:
                                print()
                                ColorOutput.info("Container will keep old port until recreated")
                                ColorOutput.print("  To recreate later:", Colors.CYAN)
                                ColorOutput.print("    [5] Stop → [4] Start", Colors.GRAY)
                                ColorOutput.print("  Or run Settings again and choose 'y'", Colors.GRAY)
            except Exception as e:
                ColorOutput.error(f"Error changing port: {e}")
        
        elif choice == '6':
            print()
            ColorOutput.print("SET MAX CONCURRENT MODELS", Colors.CYAN, bold=True)
            print()
            ColorOutput.print(f"Current limit: {self.config_manager.config.max_concurrent_models if self.config_manager.config.max_concurrent_models > 0 else 'unlimited'}", Colors.WHITE)
            print()
            ColorOutput.print("What this does:", Colors.YELLOW, bold=True)
            print("  • Limits how many models can be loaded in memory at once")
            print("  • When you load a new model and reach the limit, the oldest model is auto-unloaded")
            print("  • Helps manage memory usage, especially useful with limited RAM/VRAM")
            print()
            ColorOutput.print("Recommended values:", Colors.GRAY)
            print("  • 1 = One model at a time (auto-unload, good for low memory)")
            print("  • 2-3 = Multiple models (if you have enough RAM/VRAM)")
            print("  • 0 = Unlimited (keeps all models loaded, requires lots of memory)")
            print()
            
            try:
                new_limit = input("Enter max concurrent models (0 for unlimited): ").strip()
                if not new_limit:
                    ColorOutput.info("Setting change cancelled")
                elif not new_limit.isdigit():
                    ColorOutput.error("Invalid number")
                else:
                    limit_num = int(new_limit)
                    if limit_num < 0 or limit_num > 20:
                        ColorOutput.error("Limit must be between 0 and 20")
                    else:
                        old_limit = self.config_manager.config.max_concurrent_models
                        self.config_manager.config.max_concurrent_models = limit_num
                        self.config_manager.save_config()
                        print()
                        if limit_num == 0:
                            ColorOutput.success(f"Concurrent model limit changed from {old_limit} to unlimited!")
                            ColorOutput.print("  All loaded models will stay in memory until manually unloaded", Colors.GRAY)
                        elif limit_num == 1:
                            ColorOutput.success(f"Concurrent model limit set to {limit_num} model!")
                            ColorOutput.print("  When loading a new model, the previous one will auto-unload", Colors.GREEN)
                        else:
                            ColorOutput.success(f"Concurrent model limit changed from {old_limit} to {limit_num} models!")
                            ColorOutput.print(f"  When loading model #{limit_num + 1}, the oldest will auto-unload", Colors.GREEN)
                        print()
                        ColorOutput.info("This setting takes effect immediately for new model loads")
            except Exception as e:
                ColorOutput.error(f"Error changing limit: {e}")
        
        input("\nPress Enter to continue...")
    
    def install_model_menu(self):
        """Install a model"""
        ColorOutput.header("INSTALL MODEL")
        
        # Check if models list file exists
        models_file = Path(self.config_manager.config.models_list_file)
        
        if models_file.exists():
            try:
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                
                model_list = models_data.get('models', [])
                
                if model_list:
                    ColorOutput.print("Available models from list:", Colors.CYAN, bold=True)
                    print()
                    
                    for i, model_info in enumerate(model_list, 1):
                        model_name = model_info.get('name', 'Unknown')
                        description = model_info.get('description', '')
                        
                        # Print model number and name
                        ColorOutput.print(f"[{i}] {model_name}", Colors.WHITE, bold=True)
                        
                        # Print description with proper wrapping
                        if description:
                            # Indent description text
                            import textwrap
                            wrapped_desc = textwrap.fill(description, width=70, 
                                                        initial_indent='    ', 
                                                        subsequent_indent='    ')
                            ColorOutput.print(wrapped_desc, Colors.GRAY)
                        print()
                    
                    ColorOutput.print("─" * 60, Colors.GRAY)
                    print()
                    ColorOutput.print("Select a model by number, or type a custom model name:", Colors.WHITE)
                    choice = input("Model: ").strip()
                    
                    if choice.isdigit() and 1 <= int(choice) <= len(model_list):
                        model_name = model_list[int(choice) - 1].get('name')
                    else:
                        model_name = choice
                else:
                    ColorOutput.warning("Model list is empty")
                    model_name = input("Enter model name: ").strip()
                    
            except json.JSONDecodeError as e:
                ColorOutput.error(f"Error parsing models file: {e}")
                ColorOutput.info("The file should be in JSON format")
                model_name = input("Enter model name: ").strip()
            except Exception as e:
                ColorOutput.error(f"Error reading models file: {e}")
                model_name = input("Enter model name: ").strip()
        else:
            ColorOutput.print("Popular models:", Colors.CYAN)
            print("  • llama3.2 (3B - Fast, good for chat)")
            print("  • llama3.2:1b (1B - Very fast, basic tasks)")
            print("  • mistral (7B - Strong performance)")
            print("  • phi3 (3.8B - Microsoft, efficient)")
            print("  • qwen2.5:7b (7B - Multilingual)")
            print()
            model_name = input("Enter model name to install: ").strip()
        
        if model_name:
            self.docker.pull_model(model_name)
        
        input("\nPress Enter to continue...")
    
    def uninstall_model_menu(self):
        """Uninstall a model"""
        ColorOutput.header("UNINSTALL MODEL")
        
        models = self.docker.list_models()
        if len(models) <= 1:  # First line is header
            ColorOutput.warning("No models installed")
            input("\nPress Enter to continue...")
            return
        
        # Skip header line
        models = models[1:]
        ColorOutput.print("Installed models:", Colors.CYAN, bold=True)
        for i, model in enumerate(models, 1):
            print(f"  [{i}] {model}")
        print()
        
        choice = input("Select model number to uninstall: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_line = models[int(choice) - 1]
            # Extract model name (first column)
            model_name = model_line.split()[0]
            self.docker.remove_model(model_name)
        
        input("\nPress Enter to continue...")
    
    def list_models(self):
        """List installed models"""
        ColorOutput.header("INSTALLED MODELS")
        
        # Show where models are stored
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
    
    def view_status(self):
        """View Ollama status"""
        ColorOutput.header("OLLAMA STATUS")
        
        if not self.docker.is_docker_running():
            ColorOutput.error("Docker is not running!")
            input("\nPress Enter to continue...")
            return
        
        details = self.docker.get_container_details()
        if details:
            ColorOutput.print(f"Container Name: {details['name']}", Colors.WHITE)
            ColorOutput.print(f"Image: {details['image']}", Colors.WHITE)
            ColorOutput.print(f"Created: {details['created']}", Colors.WHITE)
            ColorOutput.print(f"Port: {details['port']}", Colors.WHITE)
            
            # Show detailed models directory information
            models_dir = self.config_manager.config.models_dir
            ColorOutput.print(f"Models Directory: {models_dir}", Colors.WHITE)
            
            # Show platform-specific info
            plat = PlatformDetector.get_platform()
            if plat == Platform.WSL:
                print()
                ColorOutput.print("ℹ️  Platform: WSL (Windows Subsystem for Linux)", Colors.CYAN)
                ColorOutput.print("   Models stored in WSL filesystem, NOT Windows", Colors.GRAY)
                ColorOutput.print("   To share models with Windows, you would need to:", Colors.GRAY)
                ColorOutput.print("   • Manually configure both to use the same path", Colors.GRAY)
                ColorOutput.print(f"   • Example: Use /mnt/c/Users/<YourName>/OllamaModels", Colors.GRAY)
            elif plat == Platform.WINDOWS:
                print()
                ColorOutput.print("ℹ️  Platform: Windows (native)", Colors.CYAN)
                ColorOutput.print("   Models stored in Windows filesystem", Colors.GRAY)
                ColorOutput.print("   If you also run this in WSL, models will be separate!", Colors.GRAY)
            elif plat == Platform.LINUX:
                print()
                ColorOutput.print("ℹ️  Platform: Linux (native)", Colors.CYAN)
            
            # Check if directory exists and show size
            if Path(models_dir).exists():
                try:
                    # Get directory size
                    total_size = sum(f.stat().st_size for f in Path(models_dir).rglob('*') if f.is_file())
                    size_gb = total_size / (1024**3)
                    print()
                    ColorOutput.print(f"   Directory exists: ✓", Colors.GREEN)
                    ColorOutput.print(f"   Storage used: {size_gb:.2f} GB", Colors.CYAN)
                except:
                    print()
                    ColorOutput.print(f"   Directory exists: ✓", Colors.GREEN)
            else:
                print()
                ColorOutput.print(f"   Directory exists: ✗ (will be created when models are installed)", Colors.YELLOW)
            
            print()
            
            if details['gpu_enabled']:
                ColorOutput.print("GPU: Enabled", Colors.GREEN)
            else:
                ColorOutput.print("GPU: Disabled", Colors.CYAN)
            
            if self.docker.container_running():
                ColorOutput.print("\nStatus: Running", Colors.GREEN, bold=True)
            else:
                ColorOutput.print("\nStatus: Stopped", Colors.YELLOW, bold=True)
        else:
            ColorOutput.warning("Container does not exist")
        
        input("\nPress Enter to continue...")
    
    def show_api_info(self):
        """Show API connection information"""
        ColorOutput.header("API CONNECTION INFO")
        
        if not self.docker.container_running():
            ColorOutput.warning("Container is not running. Start the service first.")
            input("\nPress Enter to continue...")
            return
        
        ColorOutput.success("Ollama API is running!")
        print()
        
        # Show local access
        ColorOutput.print("LOCAL ACCESS (this computer only):", Colors.CYAN, bold=True)
        print(f"  http://localhost:{self.config_manager.config.ollama_port}")
        print(f"  http://127.0.0.1:{self.config_manager.config.ollama_port}")
        print()
        
        # Show network access status
        if not self.config_manager.config.network_access:
            ColorOutput.print("NETWORK ACCESS: DISABLED ✗", Colors.GRAY, bold=True)
            print()
            ColorOutput.print("Only accessible from this computer (localhost only)", Colors.GRAY)
            print()
            ColorOutput.print("To enable network access:", Colors.CYAN)
            print("  Press [S] for Settings → [3] Enable Network Access")
            print()
            ColorOutput.print("─" * 60, Colors.GRAY)
            print()
            
            # Show basic example with localhost
            ColorOutput.print("EXAMPLE USAGE:", Colors.CYAN, bold=True)
            print()
            ColorOutput.print("Test connection:", Colors.WHITE)
            print(f"  curl http://localhost:{self.config_manager.config.ollama_port}/api/tags")
            print()
            input("\nPress Enter to continue...")
            return
        
        # Network access is enabled - show all connection methods
        ColorOutput.print("NETWORK ACCESS: ENABLED ✓", Colors.GREEN, bold=True)
        print()
        
        # Get local network IP
        local_ip = None
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            pass
        
        # Get Tailscale IP
        tailscale_ip = None
        try:
            # Try to get Tailscale IP from the interface
            result = subprocess.run(['ip', 'addr', 'show', 'tailscale0'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Parse the IP address from output
                import re
                match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', result.stdout)
                if match:
                    tailscale_ip = match.group(1)
        except:
            pass
        
        # Show local network access
        if local_ip:
            ColorOutput.print("📡 LOCAL NETWORK ACCESS (same WiFi/LAN):", Colors.YELLOW, bold=True)
            print(f"  http://{local_ip}:{self.config_manager.config.ollama_port}")
            print()
            ColorOutput.print("  Use this from:", Colors.CYAN)
            print("  • Other computers on the SAME network (home, office)")
            print("  • Your phone when connected to the SAME WiFi")
            print("  • Other devices on your local subnet")
            print()
            ColorOutput.print("  Test from another device on same network:", Colors.GRAY)
            print(f"    curl http://{local_ip}:{self.config_manager.config.ollama_port}/api/tags")
            print()
        
        # Show Tailscale access
        if tailscale_ip:
            ColorOutput.print("🔒 TAILSCALE VPN ACCESS (secure, from anywhere):", Colors.MAGENTA, bold=True)
            print(f"  http://{tailscale_ip}:{self.config_manager.config.ollama_port}")
            print()
            ColorOutput.print("  Use this from:", Colors.CYAN)
            print("  • ANY device in your Tailscale network")
            print("  • Works even on DIFFERENT networks (coffee shop, mobile data)")
            print("  • Secure encrypted connection")
            print()
            ColorOutput.print("  Requirements:", Colors.GRAY)
            print("  • Other device must have Tailscale installed")
            print("  • Other device must be logged into YOUR Tailscale network")
            print()
            ColorOutput.print("  Test from Tailscale device:", Colors.GRAY)
            print(f"    curl http://{tailscale_ip}:{self.config_manager.config.ollama_port}/api/tags")
            print()
        else:
            ColorOutput.print("🔒 TAILSCALE VPN ACCESS:", Colors.GRAY, bold=True)
            ColorOutput.print("  Tailscale not detected", Colors.GRAY)
            print()
            ColorOutput.print("  To enable Tailscale access:", Colors.CYAN)
            print("  1. Install Tailscale: https://tailscale.com/download")
            print("  2. Run: tailscale up")
            print("  3. Ollama will be accessible on your Tailscale network")
            print()
        
        # Show public WiFi warning
        if local_ip and not tailscale_ip:
            ColorOutput.print("⚠️  PUBLIC WIFI WARNING:", Colors.RED, bold=True)
            print("  On public WiFi (coffee shops, airports):")
            print("  • Local network access usually WON'T work (client isolation)")
            print("  • Use Tailscale instead for secure access")
            print()
        
        ColorOutput.print("─" * 60, Colors.GRAY)
        print()
        
        # Comprehensive examples
        ColorOutput.print("EXAMPLE USAGE:", Colors.CYAN, bold=True)
        print()
        
        # Determine best example URL
        if tailscale_ip:
            example_url = f"http://{tailscale_ip}:{self.config_manager.config.ollama_port}"
            ColorOutput.print("Using Tailscale IP (works from anywhere):", Colors.MAGENTA)
        elif local_ip:
            example_url = f"http://{local_ip}:{self.config_manager.config.ollama_port}"
            ColorOutput.print("Using local network IP:", Colors.YELLOW)
        else:
            example_url = f"http://localhost:{self.config_manager.config.ollama_port}"
            ColorOutput.print("Using localhost:", Colors.GRAY)
        print()
        
        ColorOutput.print("Test connection (curl):", Colors.WHITE)
        print(f"  curl {example_url}/api/tags")
        print()
        
        ColorOutput.print("Python example:", Colors.WHITE)
        print(f"  import requests")
        print(f"  response = requests.get('{example_url}/api/tags')")
        print(f"  print(response.json())")
        print()
        
        ColorOutput.print("Generate text:", Colors.WHITE)
        print(f'  curl {example_url}/api/generate -d \'{{')
        print(f'    "model": "llama3.2",')
        print(f'    "prompt": "Why is the sky blue?"')
        print(f'  }}\'')
        print()
        
        # Connection summary
        ColorOutput.print("CONNECTION SUMMARY:", Colors.CYAN, bold=True)
        if local_ip:
            print(f"  ✓ Local Network:  http://{local_ip}:{self.config_manager.config.ollama_port}")
        if tailscale_ip:
            print(f"  ✓ Tailscale VPN:  http://{tailscale_ip}:{self.config_manager.config.ollama_port}")
        print(f"  ✓ Localhost:      http://localhost:{self.config_manager.config.ollama_port}")
        
        input("\nPress Enter to continue...")
    
    def install_ollama(self):
        """Install Ollama (download image only)"""
        ColorOutput.header("INSTALL OLLAMA")
        
        if self.docker.image_exists():
            ColorOutput.success("Ollama image is already installed!")
            ColorOutput.print(f"Image: {self.config_manager.config.image_name}", Colors.CYAN)
            
            if not self.docker.container_exists():
                print()
                ColorOutput.info("Next step: press [2] to install a model, then [3] to start Ollama")
        else:
            ColorOutput.info("This will download the Ollama Docker image")
            ColorOutput.print("  Estimated size: ~1-2 GB", Colors.GRAY)
            print()
            
            confirm = input("Continue with installation? (y/n): ").strip().lower()
            if confirm == 'y':
                if self.docker.pull_image_with_progress():
                    print()
                    ColorOutput.success("Ollama installed successfully!")
                    ColorOutput.info("Next step: press [2] to install a model, then [3] to start Ollama")
        
        input("\nPress Enter to continue...")
    
    def chat_menu(self):
        """Chat with a model"""
        ColorOutput.header("CHAT WITH MODEL")
        
        if not self.docker.container_running():
            ColorOutput.error("Ollama container is not running")
            ColorOutput.info("Please start the container first (option [4])")
            input("\nPress Enter to continue...")
            return
        
        models = self.docker.list_models()
        if len(models) <= 1:
            ColorOutput.warning("No models installed")
            input("\nPress Enter to continue...")
            return
        
        models = models[1:]  # Skip header
        ColorOutput.print("Installed models:", Colors.CYAN, bold=True)
        for i, model in enumerate(models, 1):
            print(f"  [{i}] {model}")
        print()
        
        choice = input("Select model number to chat with: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_line = models[int(choice) - 1]
            model_name = model_line.split()[0]
            self.docker.chat_with_model(model_name)
        else:
            ColorOutput.warning("Invalid selection")
        
        input("\nPress Enter to continue...")
    
    def load_models_menu(self):
        """Load a model into memory"""
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
        
        models = models[1:]  # Skip header
        ColorOutput.print("Installed models:", Colors.CYAN, bold=True)
        for i, model in enumerate(models, 1):
            print(f"  [{i}] {model}")
        print()
        
        # Show currently loaded models
        loaded_models = self.docker.get_loaded_models()
        if loaded_models:
            ColorOutput.print("Currently loaded in memory:", Colors.YELLOW, bold=True)
            for model_info in loaded_models:
                print(f"  • {model_info['name']}")
            print()
        
        choice = input("Select model number to load into memory: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_line = models[int(choice) - 1]
            model_name = model_line.split()[0]
            self.docker.load_model(model_name)
        else:
            ColorOutput.error("Invalid selection")
        
        input("\nPress Enter to continue...")
    
    def unload_models_menu(self):
        """Unload currently loaded models from memory"""
        ColorOutput.header("UNLOAD MODELS FROM MEMORY")
        
        if not self.docker.container_running():
            ColorOutput.error("Ollama container is not running")
            input("\nPress Enter to continue...")
            return
        
        loaded_models = self.docker.get_loaded_models()
        
        if not loaded_models:
            ColorOutput.info("No models are currently loaded in memory")
            input("\nPress Enter to continue...")
            return
        
        ColorOutput.print("Currently loaded models:", Colors.CYAN, bold=True)
        for i, model_info in enumerate(loaded_models, 1):
            print(f"  [{i}] {model_info['full_line']}")
        print()
        print("  [A] Unload all models (graceful)")
        print("  [F] Force unload all (restart container)")
        print("  [0] Cancel")
        print()
        
        choice = input("Select model number to unload (or A/F for all): ").strip().upper()
        
        if choice == '0':
            ColorOutput.info("Operation cancelled")
        elif choice == 'F':
            # Force unload all by restarting container
            ColorOutput.warning("This will restart the Ollama container to clear all models")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                ColorOutput.info("Restarting container...")
                if self.docker.stop_container() and self.docker.start_container():
                    print()
                    ColorOutput.success("Container restarted - all models cleared from memory")
                else:
                    ColorOutput.error("Failed to restart container")
            else:
                ColorOutput.info("Operation cancelled")
        elif choice == 'A':
            ColorOutput.info("Unloading all models...")
            success_count = 0
            failed_models = []
            
            for model_info in loaded_models:
                if self.docker.unload_model(model_info['name'], force=False):
                    success_count += 1
                else:
                    failed_models.append(model_info['name'])
            
            print()
            ColorOutput.success(f"Unloaded {success_count} of {len(loaded_models)} models")
            
            if failed_models:
                print()
                ColorOutput.warning("Failed to unload:")
                for model in failed_models:
                    print(f"  • {model}")
                print()
                ColorOutput.print("Try force unload (option F) to restart container and clear all", Colors.YELLOW)
        elif choice.isdigit() and 1 <= int(choice) <= len(loaded_models):
            model_info = loaded_models[int(choice) - 1]
            
            # Try normal unload first
            if not self.docker.unload_model(model_info['name'], force=False):
                print()
                ColorOutput.warning("Standard unload failed")
                retry = input("Try force unload? (y/n): ").strip().lower()
                if retry == 'y':
                    self.docker.unload_model(model_info['name'], force=True)
        else:
            ColorOutput.error("Invalid selection")
        
        input("\nPress Enter to continue...")
    
    # Shared venv path — all pip installs (turboquant, requests, etc.) go here
    TURBOQUANT_VENV = VenvManager.venv_dir()

    def _install_turboquant(self) -> bool:
        """
        Install turboquant into the shared project venv (~/.ollama-manager-venv).
        Uses a venv to avoid the PEP 668 externally-managed-environment error
        on modern Debian/Ubuntu systems.  Completely isolated from the rest
        of this project and from the system Python.
        Returns True if installation succeeded.
        """
        venv_python = VenvManager.python()
        venv_pip    = VenvManager.pip()

        print()
        ColorOutput.print("=" * 65, Colors.MAGENTA, bold=True)
        ColorOutput.print("  Installing TurboQuant...", Colors.MAGENTA, bold=True)
        ColorOutput.print("=" * 65, Colors.MAGENTA, bold=True)
        print()
        ColorOutput.print(
            f"  Shared venv: {VenvManager.venv_dir()}", Colors.GRAY
        )
        ColorOutput.print(
            "  (Isolated from system Python and the rest of this project)", Colors.GRAY
        )
        print()

        # Step 1 — ensure shared venv exists (creates it if missing)
        ColorOutput.print("  Step 1/2: Ensuring shared venv exists", Colors.MAGENTA)
        ColorOutput.print(f"  Venv: {VenvManager.venv_dir()}", Colors.GRAY)
        print()
        if not VenvManager.ensure(verbose=True):
            ColorOutput.error("Could not create shared venv.")
            ColorOutput.print("  Tip: make sure you have internet access and sudo rights.", Colors.YELLOW)
            return False
        ColorOutput.success("Step 1 done.")
        print()

        # Step 2 — install turboquant into the shared venv
        ColorOutput.print("  Step 2/2: Installing turboquant into shared venv", Colors.MAGENTA)
        ColorOutput.print(f"  $ {VenvManager.pip()} install turboquant", Colors.GRAY)
        print()
        if not VenvManager.install("turboquant", verbose=True):
            return False
        ColorOutput.success("Step 2 done.")
        print()

        # Sync venv python path into TurboQuantManager
        self.turboquant.venv_python = VenvManager.python()

        print()
        ColorOutput.print("=" * 65, Colors.GREEN, bold=True)
        ColorOutput.success("TurboQuant installed successfully!")
        ColorOutput.print("=" * 65, Colors.GREEN, bold=True)
        print()
        ColorOutput.print(
            "  Use [T] TurboQuant in the main menu to start a server.", Colors.CYAN
        )
        ColorOutput.print(
            f"  Python used: {venv_python}", Colors.GRAY
        )
        print()
        return True

    def _install_nvidia_container_toolkit(self) -> bool:
        """
        Automatically install nvidia-container-toolkit on Linux/WSL.
        Runs the full 5-step process with live output.
        Returns True if installation succeeded.
        """
        print()
        ColorOutput.print("=" * 65, Colors.CYAN, bold=True)
        ColorOutput.print("  Installing nvidia-container-toolkit...", Colors.CYAN, bold=True)
        ColorOutput.print("=" * 65, Colors.CYAN, bold=True)
        print()

        steps = [
            (
                "Importing NVIDIA GPG key",
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey"
                " | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ctk-keyring.gpg",
            ),
            (
                "Adding NVIDIA apt repository",
                "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
                " | sed \'s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ctk-keyring.gpg] https://#g\'"
                " | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
            ),
            (
                "Updating apt package lists",
                "sudo apt-get update",
            ),
            (
                "Installing nvidia-container-toolkit",
                "sudo apt-get install -y nvidia-container-toolkit",
            ),
            (
                "Configuring Docker NVIDIA runtime",
                "sudo nvidia-ctk runtime configure --runtime=docker",
            ),
        ]

        for i, (label, cmd) in enumerate(steps, 1):
            ColorOutput.print(f"  Step {i}/{len(steps)}: {label}", Colors.CYAN)
            ColorOutput.print(f"  $ {cmd}", Colors.GRAY)
            print()
            try:
                result = subprocess.run(cmd, shell=True, timeout=120)
                if result.returncode != 0:
                    print()
                    ColorOutput.error(f"Step {i} failed (exit code {result.returncode})")
                    ColorOutput.print(
                        "  You may need to run the remaining steps manually.", Colors.YELLOW
                    )
                    return False
                else:
                    ColorOutput.success(f"Step {i} done.")
                    print()
            except subprocess.TimeoutExpired:
                ColorOutput.error(f"Step {i} timed out.")
                return False
            except Exception as e:
                ColorOutput.error(f"Step {i} error: {e}")
                return False

        # Restart Docker daemon (WSL-safe: try service first, then systemctl)
        ColorOutput.print("  Restarting Docker daemon...", Colors.CYAN)
        restarted = False
        for restart_cmd in ["sudo service docker restart", "sudo systemctl restart docker"]:
            try:
                r = subprocess.run(restart_cmd, shell=True, timeout=20)
                if r.returncode == 0:
                    ColorOutput.success("Docker restarted.")
                    restarted = True
                    break
            except Exception:
                pass
        if not restarted:
            ColorOutput.warning("Could not restart Docker automatically.")
            ColorOutput.print(
                "  Run manually: sudo service docker restart", Colors.CYAN
            )

        print()
        ColorOutput.print("=" * 65, Colors.GREEN, bold=True)
        ColorOutput.success("nvidia-container-toolkit installed and configured!")
        ColorOutput.print("=" * 65, Colors.GREEN, bold=True)
        print()
        ColorOutput.print(
            "  Verify GPU access with:",
            Colors.CYAN
        )
        ColorOutput.print(
            "  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi",
            Colors.WHITE
        )
        print()
        return True

    def _show_linux_requirements(self):
        """
        Display a clear requirements/prerequisites screen for Linux & WSL users.
        Shows what is needed, checks what is present, and lets the user proceed.
        GPU requirements are clearly highlighted since Ollama and TurboQuant both
        rely on the NVIDIA stack being set up correctly.
        """
        os.system('clear')

        print()
        ColorOutput.print("=" * 65, Colors.CYAN, bold=True)
        ColorOutput.print("   📋  LINUX / WSL  —  REQUIREMENTS OVERVIEW", Colors.CYAN, bold=True)
        ColorOutput.print("=" * 65, Colors.CYAN, bold=True)
        print()
        ColorOutput.print(
            "This manager runs Ollama inside Docker and can also launch",
            Colors.WHITE
        )
        ColorOutput.print(
            "TurboQuant for quantized GPU inference.  Both benefit greatly",
            Colors.WHITE
        )
        ColorOutput.print("from a properly configured NVIDIA GPU stack.", Colors.WHITE)
        print()

        # ── Section 1: Core requirements (always needed) ──────────────────────
        ColorOutput.print("─" * 65, Colors.GRAY)
        ColorOutput.print("  CORE REQUIREMENTS  (always needed)", Colors.CYAN, bold=True)
        ColorOutput.print("─" * 65, Colors.GRAY)
        print()

        # Docker
        try:
            dr = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
            docker_ok = dr.returncode == 0
            docker_ver = dr.stdout.strip() if docker_ok else ""
        except Exception:
            docker_ok = False
            docker_ver = ""

        if docker_ok:
            ColorOutput.print(f"  ✅  Docker          — {docker_ver}", Colors.GREEN)
        else:
            ColorOutput.print("  ❌  Docker          — NOT FOUND", Colors.RED)
            ColorOutput.print("      Install: sudo apt install docker.io  (Ubuntu/Debian)", Colors.GRAY)
            ColorOutput.print("      Or:      https://docs.docker.com/engine/install/", Colors.GRAY)

        # Python 3
        try:
            pr = subprocess.run(['python3', '--version'], capture_output=True, text=True, timeout=5)
            py_ok = pr.returncode == 0
            py_ver = pr.stdout.strip() if py_ok else ""
        except Exception:
            py_ok = False
            py_ver = ""

        if py_ok:
            ColorOutput.print(f"  ✅  Python 3        — {py_ver}", Colors.GREEN)
        else:
            ColorOutput.print("  ❌  Python 3        — NOT FOUND", Colors.RED)
            ColorOutput.print("      Install: sudo apt install python3", Colors.GRAY)

        # pip / requests
        try:
            rr = subprocess.run(
                ['python3', '-c', 'import requests'],
                capture_output=True, timeout=5
            )
            req_ok = rr.returncode == 0
        except Exception:
            req_ok = False

        if req_ok:
            ColorOutput.print("  ✅  requests lib    — available", Colors.GREEN)
        else:
            ColorOutput.print("  ⚠️   requests lib    — missing (needed for chat feature)", Colors.YELLOW)
            ColorOutput.print("      Auto-installed into shared venv on next startup, or run:", Colors.GRAY)
            ColorOutput.print(f"      {VenvManager.pip()} install requests", Colors.GRAY)

        print()

        # ── Section 2: GPU requirements ───────────────────────────────────────
        ColorOutput.print("─" * 65, Colors.GRAY)
        ColorOutput.print(
            "  GPU REQUIREMENTS  (for NVIDIA GPU acceleration)",
            Colors.CYAN, bold=True
        )
        ColorOutput.print(
            "  Needed for: Ollama GPU mode  •  TurboQuant inference",
            Colors.GRAY
        )
        ColorOutput.print("─" * 65, Colors.GRAY)
        print()
        ColorOutput.print(
            "  If you only have a CPU, skip this section — CPU-only mode works.",
            Colors.GRAY
        )
        print()

        # nvidia-smi
        try:
            nr = subprocess.run(['nvidia-smi'], capture_output=True, timeout=8)
            nsmi_ok = nr.returncode == 0
            if nsmi_ok:
                # grab first relevant line (GPU name)
                lines = nr.stdout.decode('utf-8', errors='ignore').splitlines()
                gpu_line = next(
                    (l for l in lines if 'GeForce' in l or 'RTX' in l or 'GTX' in l
                     or 'Quadro' in l or 'Tesla' in l or 'A100' in l or 'H100' in l),
                    ""
                ).strip()
        except Exception:
            nsmi_ok = False
            gpu_line = ""

        if nsmi_ok:
            ColorOutput.print("  ✅  nvidia-smi      — NVIDIA driver is working", Colors.GREEN)
            if gpu_line:
                ColorOutput.print(f"      GPU: {gpu_line}", Colors.GRAY)
        else:
            ColorOutput.print("  ❌  nvidia-smi      — NVIDIA driver NOT found / not working", Colors.RED)
            ColorOutput.print("      Ubuntu:  sudo apt install nvidia-driver-535", Colors.GRAY)
            ColorOutput.print("      Or use:  sudo ubuntu-drivers autoinstall", Colors.GRAY)
            if self.platform == Platform.WSL:
                ColorOutput.print(
                    "      WSL:     Install drivers in *Windows*, then restart WSL", Colors.YELLOW
                )

        # nvidia-container-toolkit
        try:
            cr = subprocess.run(
                ['which', 'nvidia-container-runtime'],
                capture_output=True, timeout=5
            )
            ctk_ok = cr.returncode == 0
        except Exception:
            ctk_ok = False

        if ctk_ok:
            ColorOutput.print("  ✅  nvidia-container-toolkit — installed", Colors.GREEN)
        else:
            ColorOutput.print("  ❌  nvidia-container-toolkit — NOT installed", Colors.RED)
            ColorOutput.print(
                "      Required so Docker containers can access your GPU.", Colors.GRAY
            )
            print()
            ColorOutput.print(
                "  Would you like to install nvidia-container-toolkit automatically?",
                Colors.CYAN
            )
            print("  This will run (with sudo):")
            ColorOutput.print("    1. Import NVIDIA GPG key", Colors.GRAY)
            ColorOutput.print("    2. Add NVIDIA apt repository", Colors.GRAY)
            ColorOutput.print("    3. apt-get install nvidia-container-toolkit", Colors.GRAY)
            ColorOutput.print("    4. nvidia-ctk runtime configure --runtime=docker", Colors.GRAY)
            print()
            ans = input("  Install now? (y/n): ").strip().lower()
            if ans == 'y':
                ctk_ok = self._install_nvidia_container_toolkit()
            else:
                ColorOutput.print("  Skipped. Manual install commands:", Colors.YELLOW)
                ColorOutput.print(
                    "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey"
                    " | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ctk-keyring.gpg",
                    Colors.GRAY
                )
                ColorOutput.print(
                    "    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
                    " | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ctk-keyring.gpg] https://#g'"
                    " | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
                    Colors.GRAY
                )
                ColorOutput.print(
                    "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit",
                    Colors.GRAY
                )
                ColorOutput.print(
                    "    sudo nvidia-ctk runtime configure --runtime=docker", Colors.GRAY
                )
            print()

        # Docker daemon NVIDIA runtime
        daemon_ok = False
        try:
            import json as _json
            daemon_path = '/etc/docker/daemon.json'
            if os.path.exists(daemon_path):
                with open(daemon_path) as f:
                    d = _json.load(f)
                    if d.get('default-runtime') == 'nvidia' or 'nvidia' in d.get('runtimes', {}):
                        daemon_ok = True
        except Exception:
            pass

        if daemon_ok:
            ColorOutput.print("  ✅  Docker NVIDIA runtime — configured in daemon.json", Colors.GREEN)
        else:
            ColorOutput.print("  ⚠️   Docker NVIDIA runtime — not configured (or unverifiable)", Colors.YELLOW)
            ColorOutput.print(
                "      Add to /etc/docker/daemon.json:", Colors.GRAY
            )
            ColorOutput.print(
                '      { "default-runtime": "nvidia", "runtimes": { "nvidia": '
                '{ "path": "nvidia-container-runtime", "runtimeArgs": [] } } }',
                Colors.GRAY
            )
            ColorOutput.print(
                "      Then: sudo systemctl restart docker", Colors.GRAY
            )

        # CUDA (host-level, informational)
        try:
            nvcc = subprocess.run(
                ['nvcc', '--version'], capture_output=True, text=True, timeout=5
            )
            cuda_ok = nvcc.returncode == 0
            cuda_ver = ""
            if cuda_ok:
                for line in nvcc.stdout.splitlines():
                    if 'release' in line.lower():
                        cuda_ver = line.strip()
                        break
        except Exception:
            cuda_ok = False
            cuda_ver = ""

        if cuda_ok:
            ColorOutput.print(f"  ✅  CUDA (nvcc)     — {cuda_ver}", Colors.GREEN)
        else:
            ColorOutput.print(
                "  ℹ️   CUDA (nvcc)     — not found on host PATH (optional for Docker use)", Colors.GRAY
            )
            ColorOutput.print(
                "      Docker images carry their own CUDA runtime.", Colors.GRAY
            )
            ColorOutput.print(
                "      TurboQuant needs CUDA libs; install via: sudo apt install nvidia-cuda-toolkit",
                Colors.GRAY
            )

        print()

        # ── Section 3: TurboQuant-specific ───────────────────────────────────
        ColorOutput.print("─" * 65, Colors.GRAY)
        ColorOutput.print(
            "  TURBOQUANT REQUIREMENTS  (only if using TurboQuant menu)",
            Colors.CYAN, bold=True
        )
        ColorOutput.print("─" * 65, Colors.GRAY)
        print()

        tq_installed = self.turboquant.is_turboquant_installed()
        if tq_installed:
            ColorOutput.print("  ✅  turboquant      — pip package installed", Colors.GREEN)
        else:
            ColorOutput.print("  ℹ️   turboquant      — not yet installed", Colors.GRAY)
            print()
            ColorOutput.print(
                "  Would you like to install TurboQuant automatically?", Colors.CYAN
            )
            print("  This will run:")
            ColorOutput.print(f"    {VenvManager.pip()} install turboquant", Colors.GRAY)
            ColorOutput.print(f"    (venv: {VenvManager.venv_dir()})", Colors.GRAY)
            print()
            ans = input("  Install now? (y/n): ").strip().lower()
            if ans == "y":
                tq_installed = self._install_turboquant()
            else:
                ColorOutput.print("  Skipped. You can install later from the [T] menu.", Colors.YELLOW)

        print()
        ColorOutput.print("─" * 65, Colors.GRAY)
        print()

        # Count issues
        issues = []
        if not docker_ok:    issues.append("Docker")
        if not nsmi_ok:      issues.append("NVIDIA driver")
        if not ctk_ok:       issues.append("nvidia-container-toolkit")

        if not issues:
            ColorOutput.print(
                "  🎉  All core requirements met! Your system is GPU-ready.",
                Colors.GREEN, bold=True
            )
        else:
            ColorOutput.print(
                f"  ⚠️  Missing: {', '.join(issues)}",
                Colors.YELLOW, bold=True
            )
            ColorOutput.print(
                "  You can still use CPU-only Ollama; GPU features won't work until fixed.",
                Colors.GRAY
            )

        print()
        ColorOutput.print(
            "  Press Enter to continue to the main menu, or Ctrl+C to exit.",
            Colors.CYAN
        )
        print()
        try:
            input()
        except KeyboardInterrupt:
            print()
            ColorOutput.info("Goodbye!")
            sys.exit(0)

    def run(self):
        """Main application loop"""
        # Check if Docker is running
        if not self.docker.is_docker_running():
            print()
            ColorOutput.print("=" * 70, Colors.RED, bold=True)
            ColorOutput.print("  ❌ DOCKER IS NOT RUNNING", Colors.RED, bold=True)
            ColorOutput.print("=" * 70, Colors.RED, bold=True)
            print()
            
            ColorOutput.warning("The Docker daemon/engine is not currently running.")
            print()
            
            # Platform-specific instructions
            plat = PlatformDetector.get_platform()
            ColorOutput.print("To fix this:", Colors.CYAN, bold=True)
            print()
            
            if plat == Platform.WINDOWS:
                ColorOutput.print("  Windows:", Colors.YELLOW, bold=True)
                print("    1. Open Docker Desktop from the Start Menu")
                print("    2. Wait for it to fully start (icon in system tray should be green)")
                print("    3. Try running this script again")
                print()
                ColorOutput.print("  If Docker Desktop is not installed:", Colors.GRAY)
                print("    • Download from: https://www.docker.com/products/docker-desktop")
            elif plat == Platform.WSL:
                ColorOutput.print("  WSL:", Colors.YELLOW, bold=True)
                print("    1. Open Docker Desktop in Windows (from Start Menu)")
                print("    2. Wait for it to fully start (icon in system tray should be green)")
                print("    3. Try running this script again")
                print()
                ColorOutput.print("  If Docker Desktop is not installed:", Colors.GRAY)
                print("    • Download from: https://www.docker.com/products/docker-desktop")
                print("    • Make sure WSL2 integration is enabled in Docker Desktop settings")
            else:  # Linux
                ColorOutput.print("  Linux:", Colors.YELLOW, bold=True)
                print("    1. Start Docker service:")
                print("       sudo systemctl start docker")
                print()
                print("    2. Or enable it to start automatically:")
                print("       sudo systemctl enable docker")
                print("       sudo systemctl start docker")
                print()
                print("    3. Try running this script again")
                print()
                ColorOutput.print("  If Docker is not installed:", Colors.GRAY)
                print("    • Install with: sudo apt install docker.io  (Ubuntu/Debian)")
                print("    • Or visit: https://docs.docker.com/engine/install/")
            
            print()
            ColorOutput.print("=" * 70, Colors.RED, bold=True)
            print()
            ColorOutput.info("Script stopped. Please start Docker and try again.")
            print()
            return
        
        # ── Linux / WSL: show requirements screen on first run ────────────────
        if self.platform in (Platform.LINUX, Platform.WSL):
            self._show_linux_requirements()
        
        while True:
            try:
                # Clear screen (cross-platform)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                self.show_main_menu()
                choice = input("Select an option: ").strip().upper()
                
                # Step 1
                if choice == '1':
                    self.install_ollama()
                # Step 2 — model management
                elif choice == '2':
                    self.install_model_menu()
                elif choice == 'D':
                    self.uninstall_model_menu()
                elif choice == 'M':
                    self.list_models()
                # Step 3 — start / stop
                elif choice == '3':
                    self.docker.start_container()
                    input("\nPress Enter to continue...")
                elif choice == '4':
                    self.docker.stop_container()
                    input("\nPress Enter to continue...")
                elif choice == '5':
                    self.view_status()
                # Use
                elif choice == '6':
                    self.chat_menu()
                elif choice == '7':
                    self.show_api_info()
                elif choice == 'L':
                    self.load_models_menu()
                elif choice == 'U':
                    self.unload_models_menu()
                # TurboQuant
                elif choice == 'T':
                    self.turboquant.show_menu()
                # Advanced
                elif choice == 'S':
                    self.handle_settings()
                elif choice == 'R':
                    self.recreate_container()
                    input("\nPress Enter to continue...")
                elif choice == '9':
                    self.docker.complete_removal()
                    input("\nPress Enter to continue...")
                elif choice == 'X':
                    self.docker.full_uninstall()
                    input("\nPress Enter to continue...")
                elif choice == '0':
                    ColorOutput.info("Goodbye!")
                    break
                
            except KeyboardInterrupt:
                print("\n")
                ColorOutput.info("Goodbye!")
                break
            except Exception as e:
                ColorOutput.error(f"An error occurred: {e}")
                input("\nPress Enter to continue...")


def main():
    """Entry point"""
    try:
        manager = OllamaManager()
        manager.run()
    except Exception as e:
        print()
        ColorOutput.error(f"FATAL ERROR: {e}")
        print()
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        print()
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()