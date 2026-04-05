"""
TurboQuant Manager for Ollama Docker Manager
Handles installation, configuration, and lifecycle of TurboQuant quantized inference server.

TurboQuant runs models directly on the host (not in Docker) using HuggingFace model IDs
with int4/int8 quantization for fast GPU inference.

Typical usage flow:
  pip install turboquant
  python3 -m turboquant.server --model <HF_MODEL_ID> --bits 4 --port 8000
  curl http://localhost:8000/health
"""

import os
import sys
import json
import subprocess
import threading
import time
import signal
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from utils import Colors, ColorOutput, PlatformDetector, Platform


# ──────────────────────────────────────────────────────────────────────────────
# Default / curated model list
# ──────────────────────────────────────────────────────────────────────────────

TURBOQUANT_SUGGESTED_MODELS: List[Dict] = [
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Qwen 2.5 3B Instruct",
        "bits": 4,
        "description": "Fast 3B multilingual model. Great all-rounder for chat and code.",
        "vram_gb": "~4 GB",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen 2.5 7B Instruct",
        "bits": 4,
        "description": "Balanced 7B model. Strong reasoning and coding.",
        "vram_gb": "~6 GB",
    },
    {
        "id": "Qwen/Qwen2.5-14B-Instruct",
        "label": "Qwen 2.5 14B Instruct",
        "bits": 4,
        "description": "Large 14B model. Excellent quality, needs ~10 GB VRAM.",
        "vram_gb": "~10 GB",
    },
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "label": "Mistral 7B Instruct v0.3",
        "bits": 4,
        "description": "Classic open-weights model. Strong instruction following.",
        "vram_gb": "~6 GB",
    },
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "label": "Llama 3.2 3B Instruct",
        "bits": 4,
        "description": "Meta's compact Llama 3.2 model. Requires HF token.",
        "vram_gb": "~4 GB",
    },
    {
        "id": "google/gemma-2-2b-it",
        "label": "Gemma 2 2B Instruct",
        "bits": 4,
        "description": "Google's lightweight Gemma 2. Requires HF token acceptance.",
        "vram_gb": "~3 GB",
    },
]

TURBOQUANT_CONFIG_FILE = "turboquant-config.json"

# Dedicated venv for TurboQuant — same path as used by OllamaManager installer
TURBOQUANT_VENV = str(Path.home() / ".turboquant-venv")


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclass (plain dict saved as JSON)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TQ_CONFIG = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "bits": 4,
    "port": 8000,
    "hf_token": "",          # optional HuggingFace token
    "last_started": "",
}


# ──────────────────────────────────────────────────────────────────────────────
# TurboQuantManager
# ──────────────────────────────────────────────────────────────────────────────

class TurboQuantManager:
    """
    Manages TurboQuant inference server lifecycle:
      - dependency installation
      - config persistence
      - server start / stop / status
      - interactive menus
    """

    def __init__(self):
        self.config = self._load_config()
        self._server_proc: Optional[subprocess.Popen] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_log_lines: List[str] = []
        # Path to venv python — set by installer, or auto-detected on init
        self.venv_python: str = self._detect_venv_python()

    # ── Config ────────────────────────────────────────────────────────────────

    def _detect_venv_python(self) -> str:
        """
        Return the venv python if ~/.turboquant-venv exists and works,
        otherwise fall back to the system python3/python.
        """
        venv_py = str(Path(TURBOQUANT_VENV) / "bin" / "python")
        if Path(venv_py).exists():
            try:
                r = subprocess.run([venv_py, "--version"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    return venv_py
            except Exception:
                pass
        return self._python_cmd()

    def _load_config(self) -> dict:
        if Path(TURBOQUANT_CONFIG_FILE).exists():
            try:
                with open(TURBOQUANT_CONFIG_FILE, "r") as f:
                    data = json.load(f)
                # Fill in any missing keys from defaults
                merged = dict(DEFAULT_TQ_CONFIG)
                merged.update(data)
                return merged
            except Exception:
                pass
        return dict(DEFAULT_TQ_CONFIG)

    def _save_config(self):
        try:
            with open(TURBOQUANT_CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            ColorOutput.warning(f"Could not save TurboQuant config: {e}")

    # ── Dependency checks ─────────────────────────────────────────────────────

    def _python_cmd(self) -> str:
        """Return 'python3' or 'python', whichever resolves."""
        for cmd in ("python3", "python"):
            try:
                r = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    return cmd
            except Exception:
                pass
        return "python3"

    def is_turboquant_installed(self) -> bool:
        """Check if turboquant is installed — checks venv first, then system pip."""
        # Check venv first
        venv_py = str(Path(TURBOQUANT_VENV) / "bin" / "python")
        if Path(venv_py).exists():
            try:
                r = subprocess.run(
                    [venv_py, "-m", "pip", "show", "turboquant"],
                    capture_output=True, text=True, timeout=10
                )
                if r.returncode == 0:
                    return True
            except Exception:
                pass
        # Fallback: check system python
        try:
            r = subprocess.run(
                [self._python_cmd(), "-m", "pip", "show", "turboquant"],
                capture_output=True, text=True, timeout=10
            )
            return r.returncode == 0
        except Exception:
            return False

    def install_turboquant(self) -> bool:
        """pip-install turboquant and fix PATH if needed."""
        ColorOutput.info("Installing turboquant via pip...")
        print()
        py = self._python_cmd()
        try:
            proc = subprocess.run(
                [py, "-m", "pip", "install", "turboquant", "--user"],
                timeout=120
            )
            if proc.returncode != 0:
                ColorOutput.error("pip install failed.")
                return False
        except subprocess.TimeoutExpired:
            ColorOutput.error("Installation timed out.")
            return False
        except Exception as e:
            ColorOutput.error(f"Installation error: {e}")
            return False

        # Ensure ~/.local/bin is on PATH for this session
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")

        ColorOutput.success("turboquant installed!")
        print()
        ColorOutput.info("To make the PATH change permanent, add this to ~/.bashrc:")
        ColorOutput.print(f"  export PATH=$HOME/.local/bin:$PATH", Colors.CYAN)
        return True

    def _check_server_health(self, port: int, timeout: int = 30) -> bool:
        """Poll /health until the server responds or timeout."""
        try:
            import urllib.request
            url = f"http://localhost:{port}/health"
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(url, timeout=2) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
                time.sleep(1)
        except Exception:
            pass
        return False

    # ── Server lifecycle ──────────────────────────────────────────────────────

    def is_server_running(self) -> bool:
        """Return True if the managed server process is alive."""
        if self._server_proc and self._server_proc.poll() is None:
            return True
        return False

    def start_server(self, model: str, bits: int, port: int, hf_token: str = "") -> bool:
        """Start the TurboQuant inference server in a background thread."""
        if self.is_server_running():
            ColorOutput.warning("TurboQuant server is already running.")
            return False

        if not self.is_turboquant_installed():
            ColorOutput.warning("TurboQuant is not installed.")
            ans = input("Install it now? (y/n): ").strip().lower()
            if ans != "y":
                return False
            if not self.install_turboquant():
                return False

        py = self.venv_python  # uses venv if installed there, else system python
        cmd = [
            py, "-m", "turboquant.server",
            "--model", model,
            "--bits", str(bits),
            "--port", str(port),
        ]

        env = os.environ.copy()
        if hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token

        ColorOutput.info(f"Starting TurboQuant server...")
        ColorOutput.print(f"  Model : {model}", Colors.WHITE)
        ColorOutput.print(f"  Bits  : {bits}-bit quantization", Colors.WHITE)
        ColorOutput.print(f"  Port  : {port}", Colors.WHITE)
        print()
        ColorOutput.print("  (First run will download the model — this may take a while)", Colors.GRAY)
        print()

        try:
            self._server_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            ColorOutput.error("Could not launch python / turboquant. Is it installed?")
            return False

        self._server_log_lines = []

        # Background thread to drain stdout so the pipe doesn't block
        def _drain():
            for line in self._server_proc.stdout:
                self._server_log_lines.append(line.rstrip())
                # Print lines that look informative (not spammy)
                stripped = line.strip()
                if stripped:
                    print(f"  {Colors.GRAY}{stripped}{Colors.RESET}")

        self._server_thread = threading.Thread(target=_drain, daemon=True)
        self._server_thread.start()

        # Save what we started
        self.config["model"] = model
        self.config["bits"] = bits
        self.config["port"] = port
        self.config["last_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_config()

        # Poll health endpoint
        ColorOutput.info(f"Waiting for server to be ready on port {port}...")
        if self._check_server_health(port, timeout=120):
            print()
            ColorOutput.success(f"TurboQuant server is ready! ✅")
            ColorOutput.print(f"  API base : http://localhost:{port}/v1", Colors.CYAN)
            ColorOutput.print(f"  Health   : http://localhost:{port}/health", Colors.CYAN)
            return True
        else:
            print()
            ColorOutput.warning("Server did not respond in time.")
            ColorOutput.print("  It may still be downloading the model or loading.", Colors.GRAY)
            ColorOutput.print(f"  Check manually: curl http://localhost:{port}/health", Colors.CYAN)
            return False

    def stop_server(self) -> bool:
        """Stop the managed server process."""
        if not self.is_server_running():
            ColorOutput.warning("No TurboQuant server is currently running (from this session).")
            return False
        ColorOutput.info("Stopping TurboQuant server...")
        try:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()
        self._server_proc = None
        ColorOutput.success("TurboQuant server stopped.")
        return True

    def kill_by_port(self, port: int) -> bool:
        """Kill whatever process is listening on the given port (Linux/WSL only)."""
        try:
            r = subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True, timeout=10
            )
            if r.returncode == 0:
                ColorOutput.success(f"Killed process on port {port}.")
                return True
            else:
                # Try lsof fallback
                lsof = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=10
                )
                pid = lsof.stdout.strip()
                if pid:
                    subprocess.run(["kill", "-9", pid], timeout=5)
                    ColorOutput.success(f"Killed PID {pid} on port {port}.")
                    return True
        except Exception as e:
            ColorOutput.warning(f"Could not kill process on port {port}: {e}")
        return False

    # ── Quick test ────────────────────────────────────────────────────────────

    def test_server(self, port: int):
        """Send a quick test prompt to the running server."""
        try:
            import urllib.request
            import json as _json

            payload = _json.dumps({
                "model": self.config.get("model", "model"),
                "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "max_tokens": 60,
            }).encode()

            req = urllib.request.Request(
                f"http://localhost:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
                msg = data["choices"][0]["message"]["content"]
                print()
                ColorOutput.success("Test response received:")
                ColorOutput.print(f'  "{msg}"', Colors.WHITE)
        except Exception as e:
            ColorOutput.error(f"Test failed: {e}")
            ColorOutput.print(
                f"  Make sure server is running: curl http://localhost:{port}/health",
                Colors.CYAN
            )

    # ── Menu helpers ──────────────────────────────────────────────────────────

    def _pick_model(self) -> Tuple[str, int]:
        """
        Interactive model picker.
        Returns (model_id, bits).
        """
        print()
        ColorOutput.print("SELECT MODEL", Colors.CYAN, bold=True)
        print()
        for i, m in enumerate(TURBOQUANT_SUGGESTED_MODELS, 1):
            vram = m["vram_gb"]
            label = m["label"]
            desc = m["description"]
            ColorOutput.print(f"  [{i}] {label}", Colors.WHITE)
            ColorOutput.print(f"      {desc}", Colors.GRAY)
            ColorOutput.print(f"      VRAM: {vram}  |  Default: {m['bits']}-bit", Colors.GRAY)
            print()
        print(f"  [C] Enter a custom HuggingFace model ID")
        print(f"  [0] Cancel")
        print()

        choice = input("Select model: ").strip().upper()
        if choice == "0":
            return "", 0
        if choice == "C":
            model_id = input("Enter HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct): ").strip()
            if not model_id:
                ColorOutput.warning("No model ID entered.")
                return "", 0
            bits_str = input("Quantization bits [4] / 8: ").strip() or "4"
            try:
                bits = int(bits_str)
                if bits not in (4, 8):
                    raise ValueError
            except ValueError:
                ColorOutput.warning("Invalid bits value, defaulting to 4.")
                bits = 4
            return model_id, bits

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(TURBOQUANT_SUGGESTED_MODELS):
                m = TURBOQUANT_SUGGESTED_MODELS[idx]
                # Allow overriding bits
                default_bits = m["bits"]
                bits_ans = input(
                    f"Quantization bits [{default_bits}] / {'8' if default_bits == 4 else '4'}: "
                ).strip() or str(default_bits)
                try:
                    bits = int(bits_ans)
                    if bits not in (4, 8):
                        raise ValueError
                except ValueError:
                    ColorOutput.warning("Invalid bits value, using default.")
                    bits = default_bits
                return m["id"], bits
        except (ValueError, IndexError):
            pass

        ColorOutput.error("Invalid selection.")
        return "", 0

    # ── Main menu ─────────────────────────────────────────────────────────────

    def show_menu(self):
        """
        Interactive TurboQuant sub-menu.
        Called from OllamaManager.
        """
        while True:
            os.system("cls" if os.name == "nt" else "clear")

            print()
            ColorOutput.print("=" * 60, Colors.MAGENTA, bold=True)
            ColorOutput.print("   ⚡ TURBOQUANT INFERENCE SERVER", Colors.MAGENTA, bold=True)
            ColorOutput.print("=" * 60, Colors.MAGENTA, bold=True)
            print()

            # Status
            installed = self.is_turboquant_installed()
            running = self.is_server_running()

            if installed:
                ColorOutput.print("  Installation : ", Colors.GRAY, end="")
                ColorOutput.print("INSTALLED", Colors.GREEN)
            else:
                ColorOutput.print("  Installation : ", Colors.GRAY, end="")
                ColorOutput.print("NOT INSTALLED", Colors.RED)

            ColorOutput.print("  Server       : ", Colors.GRAY, end="")
            if running:
                ColorOutput.print("RUNNING", Colors.GREEN)
                ColorOutput.print(
                    f"  Endpoint     : http://localhost:{self.config['port']}/v1",
                    Colors.CYAN
                )
            else:
                ColorOutput.print("STOPPED", Colors.YELLOW)

            ColorOutput.print(f"  Last model   : {self.config.get('model', '—')}", Colors.GRAY)
            ColorOutput.print(
                f"  Quant bits   : {self.config.get('bits', 4)}-bit", Colors.GRAY
            )
            ColorOutput.print(f"  Port         : {self.config.get('port', 8000)}", Colors.GRAY)

            print()
            ColorOutput.print("─" * 60, Colors.GRAY)
            print()

            ColorOutput.print("SETUP:", Colors.CYAN, bold=True)
            print("  [I] Install TurboQuant (pip)")
            print()

            ColorOutput.print("SERVER:", Colors.CYAN, bold=True)
            print("  [1] Start TurboQuant Server")
            print("  [2] Stop TurboQuant Server")
            print("  [3] Kill server on port (force)")
            print()

            ColorOutput.print("USAGE:", Colors.CYAN, bold=True)
            print("  [4] Test server (quick chat ping)")
            print("  [5] Show API info & example curl commands")
            print()

            ColorOutput.print("CONFIG:", Colors.CYAN, bold=True)
            print("  [6] Change model / bits / port")
            print("  [7] Set HuggingFace token (for gated models)")
            print()

            print("  [0] Back to main menu")
            print()
            ColorOutput.print("─" * 60, Colors.GRAY)
            print()

            choice = input("Select option: ").strip().upper()

            if choice == "0":
                break

            elif choice == "I":
                print()
                if installed:
                    ColorOutput.success("TurboQuant is already installed.")
                    ans = input("Re-install / upgrade? (y/n): ").strip().lower()
                    if ans != "y":
                        input("\nPress Enter to continue...")
                        continue
                self.install_turboquant()
                input("\nPress Enter to continue...")

            elif choice == "1":
                print()
                model = self.config.get("model", "")
                bits = self.config.get("bits", 4)
                port = self.config.get("port", 8000)
                hf_token = self.config.get("hf_token", "")

                print()
                ColorOutput.print("Use saved config or pick a new model?", Colors.CYAN)
                if model:
                    ColorOutput.print(f"  Saved: {model} ({bits}-bit, port {port})", Colors.GRAY)
                print("  [1] Use saved config")
                print("  [2] Pick a different model")
                print()
                sub = input("Choice [1]: ").strip() or "1"

                if sub == "2":
                    model, bits = self._pick_model()
                    if not model:
                        input("\nPress Enter to continue...")
                        continue
                    port_str = input(f"Port [{port}]: ").strip() or str(port)
                    try:
                        port = int(port_str)
                    except ValueError:
                        port = self.config.get("port", 8000)

                if model:
                    print()
                    self.start_server(model, bits, port, hf_token)
                input("\nPress Enter to continue...")

            elif choice == "2":
                print()
                self.stop_server()
                input("\nPress Enter to continue...")

            elif choice == "3":
                print()
                port_str = input(
                    f"Port to kill [{self.config.get('port', 8000)}]: "
                ).strip() or str(self.config.get("port", 8000))
                try:
                    port = int(port_str)
                except ValueError:
                    port = self.config.get("port", 8000)
                self.kill_by_port(port)
                input("\nPress Enter to continue...")

            elif choice == "4":
                print()
                port = self.config.get("port", 8000)
                self.test_server(port)
                input("\nPress Enter to continue...")

            elif choice == "5":
                self._show_api_info()
                input("\nPress Enter to continue...")

            elif choice == "6":
                print()
                model, bits = self._pick_model()
                if model:
                    port_str = input(
                        f"Port [{self.config.get('port', 8000)}]: "
                    ).strip() or str(self.config.get("port", 8000))
                    try:
                        port = int(port_str)
                    except ValueError:
                        port = self.config.get("port", 8000)
                    self.config["model"] = model
                    self.config["bits"] = bits
                    self.config["port"] = port
                    self._save_config()
                    print()
                    ColorOutput.success("Configuration saved.")
                input("\nPress Enter to continue...")

            elif choice == "7":
                print()
                ColorOutput.print(
                    "HuggingFace token is needed for gated models (Llama, Gemma, etc.)",
                    Colors.CYAN
                )
                ColorOutput.print(
                    "Get yours at: https://huggingface.co/settings/tokens",
                    Colors.GRAY
                )
                print()
                current = self.config.get("hf_token", "")
                if current:
                    ColorOutput.print(f"  Current token: {current[:8]}...(hidden)", Colors.GRAY)
                token = input("Enter HuggingFace token (leave blank to clear): ").strip()
                self.config["hf_token"] = token
                self._save_config()
                if token:
                    ColorOutput.success("Token saved.")
                else:
                    ColorOutput.info("Token cleared.")
                input("\nPress Enter to continue...")

            else:
                ColorOutput.error("Invalid option.")
                input("\nPress Enter to continue...")

    def _show_api_info(self):
        """Display API endpoint info and example curl commands."""
        print()
        ColorOutput.print("=" * 60, Colors.MAGENTA, bold=True)
        ColorOutput.print("  ⚡ TURBOQUANT API INFO", Colors.MAGENTA, bold=True)
        ColorOutput.print("=" * 60, Colors.MAGENTA, bold=True)
        print()

        port = self.config.get("port", 8000)
        model = self.config.get("model", "your-model-id")

        ColorOutput.print("ENDPOINTS:", Colors.CYAN, bold=True)
        print(f"  Health check : http://localhost:{port}/health")
        print(f"  Chat API     : http://localhost:{port}/v1/chat/completions")
        print(f"  OpenAI-compat: http://localhost:{port}/v1")
        print()

        ColorOutput.print("HEALTH CHECK:", Colors.CYAN, bold=True)
        ColorOutput.print(f"  curl http://localhost:{port}/health", Colors.WHITE)
        print()

        ColorOutput.print("CHAT COMPLETION:", Colors.CYAN, bold=True)
        ColorOutput.print(
            f"""  curl -X POST http://localhost:{port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{{
      "model": "{model}",
      "messages": [{{"role": "user", "content": "Hello!"}}],
      "max_tokens": 100
    }}'""",
            Colors.WHITE,
        )
        print()

        ColorOutput.print("OPENAI PYTHON SDK:", Colors.CYAN, bold=True)
        ColorOutput.print(
            f"""  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:{port}/v1", api_key="dummy")
  resp = client.chat.completions.create(
      model="{model}",
      messages=[{{"role": "user", "content": "Hello!"}}]
  )
  print(resp.choices[0].message.content)""",
            Colors.GRAY,
        )
        print()

        ColorOutput.print("COMPATIBLE WITH:", Colors.CYAN, bold=True)
        print("  • Any OpenAI-compatible client (LangChain, LlamaIndex, etc.)")
        print("  • Open WebUI (set base URL to the endpoint above)")
        print("  • Anything that supports custom OpenAI base URLs")
        print()