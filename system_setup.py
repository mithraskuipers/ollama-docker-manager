"""
Ollama Docker Manager - System Setup
Handles first-run checks, Linux/WSL requirements screen,
and automatic installation of nvidia-container-toolkit.
"""

import os
import subprocess
import sys

from utils import Platform, Colors, ColorOutput, VenvManager


class SystemSetup:
    """
    First-run system checks and optional installation helpers.

    Used by OllamaManager on Linux / WSL to verify that Docker, the NVIDIA
    driver, and nvidia-container-toolkit are present before the main menu
    is shown.
    """

    def __init__(self, platform: Platform):
        self.platform = platform

    # ── Public entry point ─────────────────────────────────────────────────────

    def show_linux_requirements(self) -> None:
        """
        Display an interactive requirements overview for Linux / WSL users.
        Offers to auto-install nvidia-container-toolkit.
        Blocks until the user presses Enter.
        """
        os.system("clear")

        print()
        ColorOutput.print("=" * 65, Colors.CYAN, bold=True)
        ColorOutput.print("   📋  LINUX / WSL  —  REQUIREMENTS OVERVIEW", Colors.CYAN, bold=True)
        ColorOutput.print("=" * 65, Colors.CYAN, bold=True)
        print()
        ColorOutput.print(
            "This manager runs Ollama inside Docker and benefits greatly",
            Colors.WHITE,
        )
        ColorOutput.print("from a properly configured NVIDIA GPU stack.", Colors.WHITE)
        print()

        self._section_core()
        self._section_gpu()

        # Summary
        ColorOutput.print("─" * 65, Colors.GRAY)
        print()
        ColorOutput.print(
            "  Press Enter to continue to the main menu, or Ctrl+C to exit.",
            Colors.CYAN,
        )
        print()
        try:
            input()
        except KeyboardInterrupt:
            print()
            ColorOutput.info("Goodbye!")
            sys.exit(0)

    # ── Requirement sections ───────────────────────────────────────────────────

    def _section_core(self) -> None:
        """Check and display core requirements (Docker, Python, requests)."""
        ColorOutput.print("─" * 65, Colors.GRAY)
        ColorOutput.print("  CORE REQUIREMENTS  (always needed)", Colors.CYAN, bold=True)
        ColorOutput.print("─" * 65, Colors.GRAY)
        print()

        # Docker
        docker_ok, docker_ver = self._check_cmd(["docker", "--version"])
        if docker_ok:
            ColorOutput.print(f"  ✅  Docker          — {docker_ver}", Colors.GREEN)
        else:
            ColorOutput.print("  ❌  Docker          — NOT FOUND", Colors.RED)
            ColorOutput.print(
                "      Install: sudo apt install docker.io  (Ubuntu/Debian)", Colors.GRAY
            )
            ColorOutput.print(
                "      Or:      https://docs.docker.com/engine/install/", Colors.GRAY
            )

        # Python 3
        py_ok, py_ver = self._check_cmd(["python3", "--version"])
        if py_ok:
            ColorOutput.print(f"  ✅  Python 3        — {py_ver}", Colors.GREEN)
        else:
            ColorOutput.print("  ❌  Python 3        — NOT FOUND", Colors.RED)
            ColorOutput.print("      Install: sudo apt install python3", Colors.GRAY)

        # requests library
        try:
            rr = subprocess.run(
                ["python3", "-c", "import requests"], capture_output=True, timeout=5
            )
            req_ok = rr.returncode == 0
        except Exception:
            req_ok = False

        if req_ok:
            ColorOutput.print("  ✅  requests lib    — available", Colors.GREEN)
        else:
            ColorOutput.print(
                "  ⚠️   requests lib    — missing (needed for chat feature)", Colors.YELLOW
            )
            ColorOutput.print(
                "      Auto-installed into shared venv on next startup, or run:", Colors.GRAY
            )
            ColorOutput.print(f"      {VenvManager.pip()} install requests", Colors.GRAY)

        print()

    def _section_gpu(self) -> None:
        """Check NVIDIA driver / container-toolkit and offer to install the toolkit."""
        ColorOutput.print("─" * 65, Colors.GRAY)
        ColorOutput.print(
            "  GPU REQUIREMENTS  (for NVIDIA GPU acceleration)", Colors.CYAN, bold=True
        )
        ColorOutput.print(
            "  Needed for: Ollama GPU mode", Colors.GRAY
        )
        ColorOutput.print("─" * 65, Colors.GRAY)
        print()
        ColorOutput.print(
            "  If you only have a CPU, skip this section — CPU-only mode works.",
            Colors.GRAY,
        )
        print()

        # nvidia-smi
        nsmi_ok = self._check_nvidia_smi()

        # nvidia-container-toolkit
        ctk_ok = self._check_container_toolkit()
        if not ctk_ok:
            print()
            ColorOutput.print(
                "  Would you like to install nvidia-container-toolkit automatically?",
                Colors.CYAN,
            )
            print("  This will run (with sudo):")
            ColorOutput.print("    1. Import NVIDIA GPG key",                 Colors.GRAY)
            ColorOutput.print("    2. Add NVIDIA apt repository",              Colors.GRAY)
            ColorOutput.print("    3. apt-get install nvidia-container-toolkit", Colors.GRAY)
            ColorOutput.print("    4. nvidia-ctk runtime configure --runtime=docker", Colors.GRAY)
            print()
            ans = input("  Install now? (y/n): ").strip().lower()
            if ans == "y":
                ctk_ok = self.install_nvidia_container_toolkit()
            else:
                ColorOutput.print("  Skipped. Manual install commands:", Colors.YELLOW)
                ColorOutput.print(
                    "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey"
                    " | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ctk-keyring.gpg",
                    Colors.GRAY,
                )
                ColorOutput.print(
                    "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit",
                    Colors.GRAY,
                )
                ColorOutput.print(
                    "    sudo nvidia-ctk runtime configure --runtime=docker", Colors.GRAY
                )
            print()

        # Docker daemon NVIDIA runtime
        self._check_daemon_runtime()

        # CUDA (informational)
        self._check_cuda()

        print()

    # ── Individual checkers ────────────────────────────────────────────────────

    @staticmethod
    def _check_cmd(cmd: list) -> tuple:
        """Run a version command. Returns (ok: bool, version_string: str)."""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return r.returncode == 0, r.stdout.strip()
        except Exception:
            return False, ""

    def _check_nvidia_smi(self) -> bool:
        try:
            nr = subprocess.run(["nvidia-smi"], capture_output=True, timeout=8)
            ok = nr.returncode == 0
            if ok:
                lines = nr.stdout.decode("utf-8", errors="ignore").splitlines()
                gpu_line = next(
                    (
                        l
                        for l in lines
                        if any(
                            k in l
                            for k in ("GeForce", "RTX", "GTX", "Quadro", "Tesla", "A100", "H100")
                        )
                    ),
                    "",
                ).strip()
                ColorOutput.print("  ✅  nvidia-smi      — NVIDIA driver is working", Colors.GREEN)
                if gpu_line:
                    ColorOutput.print(f"      GPU: {gpu_line}", Colors.GRAY)
            else:
                ColorOutput.print(
                    "  ❌  nvidia-smi      — NVIDIA driver NOT found / not working", Colors.RED
                )
                ColorOutput.print("      Ubuntu:  sudo apt install nvidia-driver-535", Colors.GRAY)
                ColorOutput.print("      Or use:  sudo ubuntu-drivers autoinstall",    Colors.GRAY)
                if self.platform == Platform.WSL:
                    ColorOutput.print(
                        "      WSL:     Install drivers in *Windows*, then restart WSL",
                        Colors.YELLOW,
                    )
            return ok
        except Exception:
            ColorOutput.print(
                "  ❌  nvidia-smi      — NVIDIA driver NOT found / not working", Colors.RED
            )
            return False

    @staticmethod
    def _check_container_toolkit() -> bool:
        try:
            cr = subprocess.run(
                ["which", "nvidia-container-runtime"], capture_output=True, timeout=5
            )
            ok = cr.returncode == 0
            if ok:
                ColorOutput.print(
                    "  ✅  nvidia-container-toolkit — installed", Colors.GREEN
                )
            else:
                ColorOutput.print(
                    "  ❌  nvidia-container-toolkit — NOT installed", Colors.RED
                )
                ColorOutput.print(
                    "      Required so Docker containers can access your GPU.", Colors.GRAY
                )
            return ok
        except Exception:
            ColorOutput.print("  ❌  nvidia-container-toolkit — NOT installed", Colors.RED)
            return False

    @staticmethod
    def _check_daemon_runtime() -> bool:
        import json as _json

        daemon_ok = False
        try:
            daemon_path = "/etc/docker/daemon.json"
            if os.path.exists(daemon_path):
                with open(daemon_path) as fh:
                    d = _json.load(fh)
                    if d.get("default-runtime") == "nvidia" or "nvidia" in d.get(
                        "runtimes", {}
                    ):
                        daemon_ok = True
        except Exception:
            pass

        if daemon_ok:
            ColorOutput.print(
                "  ✅  Docker NVIDIA runtime — configured in daemon.json", Colors.GREEN
            )
        else:
            ColorOutput.print(
                "  ⚠️   Docker NVIDIA runtime — not configured (or unverifiable)", Colors.YELLOW
            )
            ColorOutput.print(
                "      Add to /etc/docker/daemon.json:", Colors.GRAY
            )
            ColorOutput.print(
                '      { "default-runtime": "nvidia", "runtimes": { "nvidia": '
                '{ "path": "nvidia-container-runtime", "runtimeArgs": [] } } }',
                Colors.GRAY,
            )
            ColorOutput.print("      Then: sudo systemctl restart docker", Colors.GRAY)

        return daemon_ok

    @staticmethod
    def _check_cuda() -> bool:
        try:
            nvcc = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            ok = nvcc.returncode == 0
            cuda_ver = ""
            if ok:
                for line in nvcc.stdout.splitlines():
                    if "release" in line.lower():
                        cuda_ver = line.strip()
                        break
            if ok:
                ColorOutput.print(f"  ✅  CUDA (nvcc)     — {cuda_ver}", Colors.GREEN)
            else:
                ColorOutput.print(
                    "  ℹ️   CUDA (nvcc)     — not found on host PATH (optional for Docker use)",
                    Colors.GRAY,
                )
                ColorOutput.print(
                    "      Docker images carry their own CUDA runtime.", Colors.GRAY
                )
            return ok
        except Exception:
            return False

    # ── Installers ─────────────────────────────────────────────────────────────

    def install_nvidia_container_toolkit(self) -> bool:
        """
        Automatically install nvidia-container-toolkit on Linux/WSL.
        Runs the full setup process with live output.
        Returns True on success.
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
                " | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ctk-keyring.gpg] https://#g'"
                " | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
            ),
            ("Updating apt package lists",           "sudo apt-get update"),
            ("Installing nvidia-container-toolkit",   "sudo apt-get install -y nvidia-container-toolkit"),
            ("Configuring Docker NVIDIA runtime",     "sudo nvidia-ctk runtime configure --runtime=docker"),
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
                ColorOutput.success(f"Step {i} done.")
                print()
            except subprocess.TimeoutExpired:
                ColorOutput.error(f"Step {i} timed out.")
                return False
            except Exception as exc:
                ColorOutput.error(f"Step {i} error: {exc}")
                return False

        # Restart Docker daemon (WSL-safe)
        ColorOutput.print("  Restarting Docker daemon...", Colors.CYAN)
        restarted = False
        for restart_cmd in ("sudo service docker restart", "sudo systemctl restart docker"):
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
            ColorOutput.print("  Run manually: sudo service docker restart", Colors.CYAN)

        print()
        ColorOutput.print("=" * 65, Colors.GREEN, bold=True)
        ColorOutput.success("nvidia-container-toolkit installed and configured!")
        ColorOutput.print("=" * 65, Colors.GREEN, bold=True)
        print()
        ColorOutput.print("  Verify GPU access with:", Colors.CYAN)
        ColorOutput.print(
            "  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi",
            Colors.WHITE,
        )
        print()
        return True
