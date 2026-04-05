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

from utils import Colors, ColorOutput, PlatformDetector, Platform, VenvManager, poll_http_health, kill_process_on_port


# ──────────────────────────────────────────────────────────────────────────────
# Model list file  (mirrors ollama-models.json pattern)
# ──────────────────────────────────────────────────────────────────────────────

TURBOQUANT_MODELS_FILE = "turboquant-models.json"

TURBOQUANT_CONFIG_FILE = "turboquant-config.json"

# Shared venv — single source of truth is VenvManager in utils.py


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
        Return the shared venv Python if available, else fall back to system python.
        """
        if VenvManager.exists():
            venv_py = VenvManager.python()
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
        """Check if turboquant is installed in the shared venv (or system pip as fallback)."""
        # Check shared venv first
        if VenvManager.is_installed("turboquant"):
            return True
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
        """Install turboquant + required deps into the shared venv (creates venv first if needed)."""
        ColorOutput.info(f"Installing turboquant into shared venv: {VenvManager.venv_dir()}")
        print()
        if not VenvManager.ensure(verbose=True):
            ColorOutput.error("Could not create shared venv.")
            return False
        # accelerate is required by turboquant for device_map support
        for pkg in ("turboquant", "accelerate"):
            ColorOutput.info(f"Installing {pkg}...")
            if not VenvManager.install(pkg, verbose=True):
                ColorOutput.error(f"Failed to install {pkg}.")
                return False
        # Update this instance to use the venv python
        self.venv_python = VenvManager.python()
        ColorOutput.success("turboquant + accelerate installed into shared venv!")
        return True

    def _check_server_health(self, port: int, timeout: int = 30) -> bool:
        """Poll /health until the server responds or timeout. Delegates to shared utility."""
        return poll_http_health(f"http://localhost:{port}/health", timeout=timeout)

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

        # accelerate is required by transformers for device_map — auto-install if missing
        try:
            chk = subprocess.run([py, "-c", "import accelerate"], capture_output=True, timeout=10)
            if chk.returncode != 0:
                ColorOutput.warning("Missing dependency: accelerate — installing now...")
                inst = subprocess.run([py, "-m", "pip", "install", "accelerate"], timeout=180)
                if inst.returncode != 0:
                    ColorOutput.error("Failed to install accelerate. Cannot start server.")
                    return False
                ColorOutput.success("accelerate installed successfully.")
                print()
        except Exception as dep_err:
            ColorOutput.warning(f"Could not verify accelerate: {dep_err}")

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
        """Kill whatever process is listening on port. Delegates to shared utility."""
        result = kill_process_on_port(port)
        if result:
            ColorOutput.success(f"Killed process on port {port}.")
        else:
            ColorOutput.warning(f"No process found on port {port} (or could not kill it).")
        return result

    # ── Quick test ────────────────────────────────────────────────────────────

    def test_server(self, port: int):
        """
        Two-step test:
          1. Health check  — GET /health
          2. Chat ping     — POST /v1/chat/completions

        turboquant uses Python's basic http.server which is single-threaded
        and does NOT support HTTP keep-alive properly. Sending Connection: close
        causes it to BrokenPipe on the *next* request. The correct approach is
        to let the server manage the connection itself — open, read fully, done.
        We use a fresh socket per request (requests does this automatically when
        you don't reuse a Session across calls) and never send Connection: close.
        """
        try:
            import requests as _req
        except ImportError:
            ColorOutput.error("'requests' library not found.")
            ColorOutput.print("  Install with: pip install requests", Colors.CYAN)
            return

        model = self.config.get("model", "model")
        base  = f"http://localhost:{port}"

        # ── Step 1: health ────────────────────────────────────────────────────
        print()
        ColorOutput.info("Checking server health...")
        try:
            # (connect_timeout, read_timeout) — read up to 10 s for the body
            hr = _req.get(f"{base}/health", timeout=(5, 10))
            if hr.status_code == 200:
                ColorOutput.success(f"Health endpoint OK  (HTTP {hr.status_code})")
            else:
                ColorOutput.warning(f"Health endpoint returned HTTP {hr.status_code}")
        except _req.exceptions.ConnectionError:
            ColorOutput.error("Cannot connect — is the server running?")
            ColorOutput.print(f"  Try: curl {base}/health", Colors.CYAN)
            return
        except _req.exceptions.Timeout:
            ColorOutput.error("Health check timed out — server may be busy loading the model.")
            return
        except Exception as e:
            ColorOutput.error(f"Health check failed: {e}")
            return

        # ── Step 2: chat ping ─────────────────────────────────────────────────
        print()
        ColorOutput.info(f"Sending test prompt to {model}...")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly one sentence: Hello!"}],
            "max_tokens": 60,
        }
        try:
            # Do NOT reuse a Session and do NOT send Connection: close.
            # turboquant's http.server handles one request per connection;
            # a fresh request() call opens a fresh socket automatically.
            resp = _req.post(
                f"{base}/v1/chat/completions",
                json=payload,
                timeout=(5, 90),   # 5s connect, 90s read (model inference can be slow)
            )
            if resp.status_code == 200:
                data = resp.json()
                msg  = data["choices"][0]["message"]["content"].strip()
                print()
                ColorOutput.success("Response received:")
                ColorOutput.print(f'  "{msg}"', Colors.WHITE)
            else:
                ColorOutput.error(f"Chat ping failed  (HTTP {resp.status_code})")
                ColorOutput.print(f"  Body: {resp.text[:300]}", Colors.GRAY)
        except _req.exceptions.Timeout:
            ColorOutput.error("Chat ping timed out — model may still be warming up, try again.")
        except Exception as e:
            ColorOutput.error(f"Chat ping failed: {e}")

    # ── Menu helpers ──────────────────────────────────────────────────────────

    def _load_models_list(self) -> List[Dict]:
        """
        Load the curated model list from turboquant-models.json.
        Falls back to an empty list if the file is missing or malformed —
        the user can still enter a custom HuggingFace ID.
        """
        models_path = Path(TURBOQUANT_MODELS_FILE)
        if not models_path.exists():
            ColorOutput.warning(
                f"Model list file not found: {TURBOQUANT_MODELS_FILE}"
            )
            ColorOutput.print(
                "  Place turboquant-models.json next to this script to enable the model list.",
                Colors.GRAY,
            )
            return []
        try:
            with open(models_path, "r") as fh:
                data = json.load(fh)
            return data.get("models", [])
        except json.JSONDecodeError as exc:
            ColorOutput.error(f"Could not parse {TURBOQUANT_MODELS_FILE}: {exc}")
            return []
        except Exception as exc:
            ColorOutput.error(f"Could not read {TURBOQUANT_MODELS_FILE}: {exc}")
            return []

    def _pick_model(self) -> Tuple[str, int]:
        """
        Interactive model picker.  Reads from turboquant-models.json —
        the same pattern as ollama-models.json / menu_handler.install_model().
        Returns (model_id, bits) or ("", 0) on cancel.
        """
        import textwrap

        model_list = self._load_models_list()

        print()
        ColorOutput.print("SELECT TURBOQUANT MODEL", Colors.CYAN, bold=True)
        print()

        if model_list:
            for i, m in enumerate(model_list, 1):
                label   = m.get("label") or m.get("id", "Unknown")
                hf_id   = m.get("id", "")
                desc    = m.get("description", "")
                vram    = m.get("vram_gb", "?")
                bits    = m.get("bits", 4)

                ColorOutput.print(f"  [{i}] {label}", Colors.WHITE, bold=True)
                ColorOutput.print(f"      {hf_id}", Colors.GRAY)
                if desc:
                    wrapped = textwrap.fill(
                        desc, width=70,
                        initial_indent="      ",
                        subsequent_indent="      ",
                    )
                    ColorOutput.print(wrapped, Colors.GRAY)
                ColorOutput.print(
                    f"      VRAM: {vram}  |  Default quantization: {bits}-bit",
                    Colors.GRAY,
                )
                print()
        else:
            ColorOutput.print(
                "  No models in list — enter a custom HuggingFace ID below.",
                Colors.YELLOW,
            )
            print()

        ColorOutput.print("─" * 60, Colors.GRAY)
        print()
        print("  [C] Enter a custom HuggingFace model ID")
        print("  [0] Cancel")
        print()

        choice = input("Select model: ").strip().upper()

        if choice == "0":
            return "", 0

        if choice == "C" or (not model_list and choice != "0"):
            model_id = input(
                "HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct): "
            ).strip()
            if not model_id:
                ColorOutput.warning("No model ID entered.")
                return "", 0
            bits = self._ask_bits(4)
            return model_id, bits

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_list):
                m            = model_list[idx]
                default_bits = m.get("bits", 4)
                bits         = self._ask_bits(default_bits)
                return m["id"], bits
        except (ValueError, IndexError):
            pass

        ColorOutput.error("Invalid selection.")
        return "", 0

    def _ask_bits(self, default: int = 4) -> int:
        """Prompt for quantization bits (4 or 8), returning the default on invalid input."""
        other = 8 if default == 4 else 4
        raw = input(f"Quantization bits [{default}] / {other}: ").strip()
        if not raw:
            return default
        try:
            bits = int(raw)
            if bits not in (4, 8):
                raise ValueError
            return bits
        except ValueError:
            ColorOutput.warning(f"Invalid value — using {default}-bit.")
            return default

    # ── Main menu ─────────────────────────────────────────────────────────────

    def show_menu(self):
        """
        Interactive TurboQuant sub-menu.
        Called from OllamaManager when the user selects [T].

        ⚠  TurboQuant uses HuggingFace Transformers and downloads its own
           model weights from huggingface.co.  These are DIFFERENT files from
           the GGUF models used by Ollama — both can coexist on the same machine.
        """
        while True:
            os.system("cls" if os.name == "nt" else "clear")

            installed = self.is_turboquant_installed()
            running   = self.is_server_running()
            model     = self.config.get("model", "—")
            bits      = self.config.get("bits", 4)
            port      = self.config.get("port", 8000)
            hf_token  = self.config.get("hf_token", "")

            # ── Header ───────────────────────────────────────────────────────
            ColorOutput.print("=" * 60, Colors.MAGENTA, bold=True)
            ColorOutput.print("   ⚡ TURBOQUANT  —  GPU Quantized Inference", Colors.MAGENTA, bold=True)
            ColorOutput.print("=" * 60, Colors.MAGENTA, bold=True)
            print()

            # ── Status panel ─────────────────────────────────────────────────
            inst_str  = f"{Colors.GREEN}INSTALLED{Colors.RESET}"   if installed else f"{Colors.RED}NOT INSTALLED{Colors.RESET}"
            srv_str   = f"{Colors.GREEN}RUNNING{Colors.RESET}"     if running   else f"{Colors.YELLOW}STOPPED{Colors.RESET}"
            token_str = f"{Colors.GREEN}set{Colors.RESET}" if hf_token else f"{Colors.GRAY}not set{Colors.RESET}"

            print(f"  Package  : {inst_str}")
            print(f"  Server   : {srv_str}")
            if running:
                ColorOutput.print(f"  Endpoint : http://localhost:{port}/v1", Colors.CYAN)
            print(f"  Model    : {Colors.WHITE}{model}{Colors.RESET}  ({bits}-bit)")
            print(f"  Port     : {port}")
            print(f"  HF Token : {token_str}")
            print()

            # ── Important note about model storage ───────────────────────────
            ColorOutput.print("  ℹ  TurboQuant downloads its own HuggingFace model weights.", Colors.YELLOW)
            ColorOutput.print("     These are separate from your Ollama/GGUF models.", Colors.GRAY)
            ColorOutput.print("     First launch will download the model (~2–15 GB).", Colors.GRAY)
            print()
            ColorOutput.print("─" * 60, Colors.GRAY)
            print()

            # ── Menu items — context-sensitive ───────────────────────────────
            if not installed:
                ColorOutput.print("  SETUP  (required before first use)", Colors.CYAN, bold=True)
                print("    [I] Install TurboQuant  (pip install turboquant accelerate)")
                print()
            else:
                ColorOutput.print("  SERVER", Colors.CYAN, bold=True)
                if not running:
                    print("    [1] Start Server      (launches inference on the configured model)")
                    print("    [2] Start with a different model")
                else:
                    print("    [3] Stop Server")
                    print("    [4] Force-kill server on port  (if Stop doesn't work)")
                print()

                ColorOutput.print("  TEST & INSPECT", Colors.CYAN, bold=True)
                print("    [5] Send a test message  (quick health + chat ping)")
                print("    [6] Show API endpoints & curl examples")
                print()

                ColorOutput.print("  CONFIGURE", Colors.CYAN, bold=True)
                print("    [7] Change model, quantization bits, or port")
                print("    [8] Set HuggingFace token  (required for Llama, Gemma, etc.)")
                print()

                ColorOutput.print("  SETUP", Colors.CYAN, bold=True)
                print("    [I] Re-install / upgrade TurboQuant")
                print()

            print("    [0] Back to main menu")
            print()
            ColorOutput.print("─" * 60, Colors.GRAY)
            print()

            choice = input("Select option: ").strip().upper()

            # ── 0 — back ─────────────────────────────────────────────────────
            if choice == "0":
                break

            # ── I — install ──────────────────────────────────────────────────
            elif choice == "I":
                print()
                if installed:
                    ColorOutput.success("TurboQuant is already installed.")
                    ans = input("Re-install / upgrade? (y/n): ").strip().lower()
                    if ans != "y":
                        continue
                self.install_turboquant()
                input("\nPress Enter to continue...")

            # ── 1 — start with saved config ──────────────────────────────────
            elif choice == "1" and installed and not running:
                if not model or model == "—":
                    ColorOutput.warning("No model configured yet. Use [2] to pick one first.")
                    input("\nPress Enter to continue...")
                    continue
                print()
                ColorOutput.print(f"Starting: {model}  ({bits}-bit)  on port {port}", Colors.CYAN)
                print()
                self.start_server(model, bits, port, hf_token)
                input("\nPress Enter to continue...")

            # ── 2 — pick a different model then start ─────────────────────────
            elif choice == "2" and installed and not running:
                print()
                new_model, new_bits = self._pick_model()
                if not new_model:
                    continue
                port_str = input(f"Port [{port}]: ").strip() or str(port)
                try:
                    port = int(port_str)
                except ValueError:
                    port = self.config.get("port", 8000)
                print()
                self.start_server(new_model, new_bits, port, hf_token)
                input("\nPress Enter to continue...")

            # ── 3 — stop ─────────────────────────────────────────────────────
            elif choice == "3" and installed and running:
                print()
                self.stop_server()
                input("\nPress Enter to continue...")

            # ── 4 — force kill ────────────────────────────────────────────────
            elif choice == "4" and installed and running:
                print()
                self.kill_by_port(port)
                self._server_proc = None
                input("\nPress Enter to continue...")

            # ── 5 — test ─────────────────────────────────────────────────────
            elif choice == "5" and installed:
                print()
                if not running:
                    ColorOutput.warning("Server is not running. Start it first with [1] or [2].")
                else:
                    self.test_server(port)
                input("\nPress Enter to continue...")

            # ── 6 — API info ─────────────────────────────────────────────────
            elif choice == "6" and installed:
                self._show_api_info()
                input("\nPress Enter to continue...")

            # ── 7 — change config ─────────────────────────────────────────────
            elif choice == "7" and installed:
                print()
                new_model, new_bits = self._pick_model()
                if new_model:
                    port_str = input(
                        f"Port [{self.config.get('port', 8000)}]: "
                    ).strip() or str(self.config.get("port", 8000))
                    try:
                        new_port = int(port_str)
                    except ValueError:
                        new_port = self.config.get("port", 8000)
                    self.config["model"] = new_model
                    self.config["bits"]  = new_bits
                    self.config["port"]  = new_port
                    self._save_config()
                    print()
                    ColorOutput.success("Configuration saved.")
                    if running:
                        ColorOutput.warning("Server is running with the old model. Restart it to apply changes.")
                input("\nPress Enter to continue...")

            # ── 8 — HF token ─────────────────────────────────────────────────
            elif choice == "8" and installed:
                print()
                ColorOutput.print("HuggingFace token — required for gated models (Llama, Gemma, Mistral…)", Colors.CYAN)
                ColorOutput.print("Get yours at: https://huggingface.co/settings/tokens", Colors.GRAY)
                print()
                if hf_token:
                    ColorOutput.print(f"  Current token: {hf_token[:8]}••••••••  (hidden)", Colors.GRAY)
                token = input("Enter token (leave blank to clear): ").strip()
                self.config["hf_token"] = token
                self._save_config()
                print()
                ColorOutput.success("Token saved.") if token else ColorOutput.info("Token cleared.")
                input("\nPress Enter to continue...")

            else:
                ColorOutput.error("Invalid option or action not available in current state.")
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