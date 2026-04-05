"""
Utility classes for Ollama Docker Manager
Contains platform detection, color output, and basic configuration
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class Platform(Enum):
    """Supported platforms"""
    WINDOWS = "Windows"
    LINUX = "Linux"
    WSL = "WSL"
    UNKNOWN = "Unknown"


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


@dataclass
class OllamaConfig:
    """Configuration for Ollama Docker Manager"""
    container_name: str = "ollama"
    image_name: str = "ollama/ollama"
    models_list_file: str = "ollama-models.json"  # Changed from .txt to .json
    ollama_port: int = 11434
    volume_name: str = "ollama"
    use_gpu: bool = False
    network_access: bool = False  # Allow network access (0.0.0.0) vs localhost only
    config_file: str = "ollama-config.json"
    models_dir: str = ""  # Will be set based on platform
    max_concurrent_models: int = 1  # Maximum number of models to keep loaded in memory


class PlatformDetector:
    """Detect and handle platform-specific operations"""
    
    @staticmethod
    def get_platform() -> Platform:
        """Detect the current platform"""
        system = platform.system()
        
        if system == "Windows":
            return Platform.WINDOWS
        elif system == "Linux":
            # Check if running in WSL
            try:
                with open('/proc/version', 'r') as f:
                    if 'microsoft' in f.read().lower():
                        return Platform.WSL
            except:
                pass
            return Platform.LINUX
        else:
            return Platform.UNKNOWN
    
    @staticmethod
    def get_models_directory() -> str:
        r"""
        Get the platform-specific models directory path
        
        IMPORTANT PLATFORM NOTES:
        - Windows native: C:\Users\YourName\OllamaModels
        - Linux: /home/yourname/OllamaModels  
        - WSL: /home/yourname/OllamaModels (WSL filesystem, NOT Windows)
        
        These are SEPARATE directories! Running on Windows native and WSL
        will use different storage locations and models will NOT be shared
        unless you manually configure the paths to be the same.
        """
        plat = PlatformDetector.get_platform()
        
        if plat == Platform.WINDOWS:
            # Windows native - uses Windows user home directory
            # Example: C:\Users\YourName\OllamaModels
            home = Path.home()
            models_dir = home / "OllamaModels"
        elif plat in [Platform.LINUX, Platform.WSL]:
            # Linux or WSL - uses Linux/WSL user home directory
            # WSL example: /home/yourname/OllamaModels (not /mnt/c/Users/...)
            # Linux example: /home/yourname/OllamaModels
            home = Path.home()
            models_dir = home / "OllamaModels"
        else:
            # Fallback for unknown platforms
            models_dir = Path.home() / "OllamaModels"
        
        return str(models_dir)
    
    @staticmethod
    def ensure_models_directory() -> Tuple[bool, str]:
        """
        Create and verify the models directory exists
        Returns: (success, directory_path)
        """
        models_dir = PlatformDetector.get_models_directory()
        
        try:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            
            # Verify it's writable
            test_file = Path(models_dir) / ".test_write"
            test_file.touch()
            test_file.unlink()
            
            return True, models_dir
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def supports_gpu() -> bool:
        """Check if GPU is available on this platform"""
        result = PlatformDetector.get_gpu_diagnostics()
        return result['available']
    
    @staticmethod
    def get_gpu_diagnostics() -> Dict:
        """Get detailed GPU diagnostics"""
        plat = PlatformDetector.get_platform()
        
        if plat == Platform.WINDOWS:
            return PlatformDetector._check_windows_gpu()
        elif plat in [Platform.LINUX, Platform.WSL]:
            return PlatformDetector._check_linux_gpu()
        return {
            'available': False,
            'gpu_detected': False,
            'driver_installed': False,
            'docker_support': False,
            'issues': ['Unknown platform'],
            'recommendations': []
        }
    
    @staticmethod
    def _check_windows_gpu() -> Dict:
        """Check GPU availability on Windows with detailed diagnostics"""
        result = {
            'available': False,
            'gpu_detected': False,
            'driver_installed': False,
            'docker_support': False,
            'wsl_configured': False,
            'issues': [],
            'recommendations': []
        }
        
        # Check 1: NVIDIA GPU exists (using PowerShell for Windows 11 compatibility)
        try:
            # Try PowerShell first (works on all Windows versions)
            gpu_result = subprocess.run(
                ['powershell', '-Command', 
                 "Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' } | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=10
            )
            
            if gpu_result.returncode == 0 and gpu_result.stdout.strip():
                result['gpu_detected'] = True
                result['gpu_name'] = gpu_result.stdout.strip().split('\n')[0]
            else:
                result['issues'].append('No NVIDIA GPU detected in Windows')
                result['recommendations'].append('⚠ This feature requires an NVIDIA GPU (GeForce, RTX, Quadro, etc.)')
                result['recommendations'].append('  If you have an NVIDIA GPU but it\'s not detected, check Device Manager')
                return result
        except Exception as e:
            result['issues'].append(f'Could not check for GPU: {e}')
            result['recommendations'].append('⚠ Unable to detect GPU hardware')
            result['recommendations'].append('  Try opening Device Manager to verify your GPU is recognized')
            return result
        
        # Check 2: Windows NVIDIA driver
        try:
            nvidia_smi_result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True, timeout=5
            )
            if nvidia_smi_result.returncode == 0:
                result['driver_installed'] = True
            else:
                result['issues'].append('NVIDIA driver not found in Windows')
                result['recommendations'].append('')
                result['recommendations'].append('📥 Install NVIDIA GPU drivers:')
                result['recommendations'].append('  1. Visit: https://www.nvidia.com/download/index.aspx')
                result['recommendations'].append('  2. Select your GPU model')
                result['recommendations'].append('  3. Download and install the latest driver')
                result['recommendations'].append('  4. Restart your computer')
                return result
        except FileNotFoundError:
            result['issues'].append('nvidia-smi command not found')
            result['recommendations'].append('')
            result['recommendations'].append('📥 Install NVIDIA GPU drivers:')
            result['recommendations'].append('  1. Visit: https://www.nvidia.com/download/index.aspx')
            result['recommendations'].append('  2. Select your GPU model')
            result['recommendations'].append('  3. Download and install the latest driver')
            result['recommendations'].append('  4. Restart your computer')
            return result
        
        # Check 3: WSL2 NVIDIA driver (for Docker Desktop GPU access)
        # Docker Desktop on Windows uses WSL2 backend, which needs special NVIDIA drivers
        try:
            # Check if WSL is installed
            wsl_check = subprocess.run(
                ['wsl', '--status'],
                capture_output=True, text=True, timeout=5
            )
            
            if wsl_check.returncode == 0:
                # WSL is installed, check for CUDA support
                cuda_check = subprocess.run(
                    ['wsl', 'nvidia-smi'],
                    capture_output=True, timeout=5
                )
                
                if cuda_check.returncode == 0:
                    result['wsl_configured'] = True
                    result['docker_support'] = True
                    result['available'] = True
                else:
                    result['issues'].append('WSL2 CUDA support not configured')
                    result['recommendations'].append('')
                    result['recommendations'].append('🔧 Enable GPU in WSL2 for Docker:')
                    result['recommendations'].append('  1. Install WSL2 CUDA drivers from NVIDIA:')
                    result['recommendations'].append('     https://docs.nvidia.com/cuda/wsl-user-guide/index.html')
                    result['recommendations'].append('  2. In Docker Desktop:')
                    result['recommendations'].append('     Settings → Resources → WSL Integration')
                    result['recommendations'].append('     Enable integration for your WSL distributions')
                    result['recommendations'].append('  3. Restart Docker Desktop')
            else:
                result['issues'].append('WSL2 not properly configured')
                result['recommendations'].append('')
                result['recommendations'].append('🔧 GPU acceleration in Docker requires WSL2:')
                result['recommendations'].append('  1. Enable WSL2: wsl --install')
                result['recommendations'].append('  2. Install WSL2 CUDA drivers from:')
                result['recommendations'].append('     https://docs.nvidia.com/cuda/wsl-user-guide/index.html')
                result['recommendations'].append('  3. Enable WSL integration in Docker Desktop')
        except FileNotFoundError:
            result['issues'].append('WSL not installed')
            result['recommendations'].append('')
            result['recommendations'].append('🔧 GPU acceleration in Docker requires WSL2:')
            result['recommendations'].append('  1. Install WSL2: wsl --install')
            result['recommendations'].append('  2. Install WSL2 CUDA drivers from:')
            result['recommendations'].append('     https://docs.nvidia.com/cuda/wsl-user-guide/index.html')
            result['recommendations'].append('  3. Enable WSL integration in Docker Desktop')
        
        return result
    
    @staticmethod
    def _check_linux_gpu() -> Dict:
        """Check GPU availability on Linux/WSL"""
        result = {
            'available': False,
            'gpu_detected': False,
            'driver_installed': False,
            'docker_support': False,
            'daemon_configured': False,
            'issues': [],
            'recommendations': []
        }
        
        plat = PlatformDetector.get_platform()
        is_wsl = (plat == Platform.WSL)
        
        # Check 1: nvidia-smi exists and works
        try:
            nvidia_result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True, timeout=5
            )
            
            if nvidia_result.returncode == 0:
                result['gpu_detected'] = True
                result['driver_installed'] = True
            else:
                result['issues'].append('NVIDIA driver not working')
                if is_wsl:
                    result['recommendations'].append('🔧 WSL GPU Setup:')
                    result['recommendations'].append('  1. Install NVIDIA drivers in Windows (not WSL)')
                    result['recommendations'].append('     Visit: https://www.nvidia.com/download/index.aspx')
                    result['recommendations'].append('  2. In Windows PowerShell: wsl --shutdown')
                    result['recommendations'].append('  3. Restart WSL and test: nvidia-smi')
                else:
                    result['recommendations'].append('Install NVIDIA drivers:')
                    result['recommendations'].append('  Ubuntu/Debian: sudo apt install nvidia-driver-535')
                    result['recommendations'].append('  Or use: ubuntu-drivers devices && sudo ubuntu-drivers autoinstall')
                return result
        except FileNotFoundError:
            result['issues'].append('nvidia-smi not found')
            if is_wsl:
                result['recommendations'].append('Install NVIDIA drivers in Windows first')
                result['recommendations'].append('  Visit: https://www.nvidia.com/download/index.aspx')
            else:
                result['recommendations'].append('Install NVIDIA drivers first')
            return result
        
        # Check 2: Docker daemon configuration for NVIDIA runtime
        daemon_config_path = '/etc/docker/daemon.json'
        try:
            import json
            daemon_configured = False
            
            if os.path.exists(daemon_config_path):
                with open(daemon_config_path, 'r') as f:
                    daemon_config = json.load(f)
                    # Check if nvidia runtime is configured
                    if daemon_config.get('default-runtime') == 'nvidia' or \
                       'nvidia' in daemon_config.get('runtimes', {}):
                        daemon_configured = True
                        result['daemon_configured'] = True
            
            if not daemon_configured:
                result['issues'].append('Docker daemon not configured for NVIDIA runtime')
        except Exception:
            result['issues'].append('Could not verify Docker daemon configuration')
        
        # Check 3: nvidia-container-toolkit and Docker GPU support
        try:
            # First check if nvidia-container-runtime is installed
            runtime_check = subprocess.run(
                ['which', 'nvidia-container-runtime'],
                capture_output=True, timeout=5
            )
            
            if runtime_check.returncode != 0:
                result['issues'].append('nvidia-container-toolkit not installed')
                result['recommendations'].append('')
                result['recommendations'].append('📦 Install NVIDIA Container Toolkit:')
                result['recommendations'].append('  1. Setup repository:')
                result['recommendations'].append('     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg')
                result['recommendations'].append('     curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\')
                result['recommendations'].append('       sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \\')
                result['recommendations'].append('       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list')
                result['recommendations'].append('  2. Install:')
                result['recommendations'].append('     sudo apt-get update')
                result['recommendations'].append('     sudo apt-get install -y nvidia-container-toolkit')
                result['recommendations'].append('  3. Configure Docker daemon:')
                result['recommendations'].append('     sudo nvidia-ctk runtime configure --runtime=docker')
                result['recommendations'].append('  4. CRITICAL - Restart Docker:')
                result['recommendations'].append('     sudo systemctl restart docker')
                if is_wsl:
                    result['recommendations'].append('     Or: sudo service docker restart')
                result['recommendations'].append('')
                result['recommendations'].append('  5. Verify GPU access:')
                result['recommendations'].append('     docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi')
                return result
            
            # Now test actual Docker GPU support
            docker_check = subprocess.run(
                ['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.8.0-base-ubuntu22.04', 'nvidia-smi'],
                capture_output=True, timeout=15
            )
            
            if docker_check.returncode == 0:
                result['docker_support'] = True
                result['available'] = True
            else:
                # Container toolkit is installed but not working
                error_msg = docker_check.stderr.decode('utf-8', errors='ignore').lower()
                
                if 'could not select device driver' in error_msg or 'unknown runtime' in error_msg or 'nvidia' in error_msg:
                    result['issues'].append('Docker daemon not configured for NVIDIA runtime')
                    result['recommendations'].append('')
                    result['recommendations'].append('🔧 Configure Docker for GPU (CRITICAL STEPS):')
                    result['recommendations'].append('  1. Create/edit /etc/docker/daemon.json with this content:')
                    result['recommendations'].append('     {')
                    result['recommendations'].append('       "default-runtime": "nvidia",')
                    result['recommendations'].append('       "runtimes": {')
                    result['recommendations'].append('         "nvidia": {')
                    result['recommendations'].append('           "path": "nvidia-container-runtime",')
                    result['recommendations'].append('           "runtimeArgs": []')
                    result['recommendations'].append('         }')
                    result['recommendations'].append('       }')
                    result['recommendations'].append('     }')
                    result['recommendations'].append('')
                    result['recommendations'].append('  2. CRITICAL - Restart Docker daemon:')
                    result['recommendations'].append('     sudo systemctl restart docker')
                    if is_wsl:
                        result['recommendations'].append('     Or: sudo service docker restart')
                        result['recommendations'].append('     Or in Windows PowerShell: wsl --shutdown (then restart WSL)')
                    result['recommendations'].append('')
                    result['recommendations'].append('  3. Verify Docker sees NVIDIA runtime:')
                    result['recommendations'].append('     docker info | grep -i runtime')
                    result['recommendations'].append('')
                    result['recommendations'].append('  4. Test GPU access:')
                    result['recommendations'].append('     docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi')
                else:
                    result['issues'].append('Docker GPU test failed')
                    result['recommendations'].append('Ensure nvidia-container-toolkit is properly installed and Docker is restarted')
                    result['recommendations'].append('  sudo systemctl restart docker')
                    if is_wsl:
                        result['recommendations'].append('  Or: sudo service docker restart')
                
        except Exception as e:
            result['issues'].append(f'Could not test Docker GPU support: {e}')
            result['recommendations'].append('Ensure nvidia-container-toolkit is installed and Docker is restarted')
        
        return result


class ColorOutput:
    """Helper class for colored terminal output"""
    
    @staticmethod
    def print(text: str, color: str = Colors.WHITE, bold: bool = False, end: str = '\n'):
        """Print colored text"""
        if bold:
            print(f"{Colors.BOLD}{color}{text}{Colors.RESET}", end=end)
        else:
            print(f"{color}{text}{Colors.RESET}", end=end)
    
    @staticmethod
    def success(text: str):
        """Print success message"""
        ColorOutput.print(f"✓ {text}", Colors.GREEN)
    
    @staticmethod
    def error(text: str):
        """Print error message"""
        ColorOutput.print(f"✗ {text}", Colors.RED)
    
    @staticmethod
    def warning(text: str):
        """Print warning message"""
        ColorOutput.print(f"⚠ {text}", Colors.YELLOW)
    
    @staticmethod
    def info(text: str):
        """Print info message"""
        ColorOutput.print(f"ℹ {text}", Colors.CYAN)
    
    @staticmethod
    def header(text: str):
        """Print section header"""
        print()
        ColorOutput.print("=" * 60, Colors.CYAN)
        ColorOutput.print(f"  {text}", Colors.CYAN, bold=True)
        ColorOutput.print("=" * 60, Colors.CYAN)
        print()


@dataclass
class CommandResult:
    """Result of a command execution"""
    returncode: int
    stdout: str
    stderr: str


# ──────────────────────────────────────────────────────────────────────────────
# Shared venv manager
# Single source of truth for the project-wide virtual environment.
# All pip installs (turboquant, requests, etc.) go here so there is
# never a conflict with the system Python or with each other.
# ──────────────────────────────────────────────────────────────────────────────

import sys as _sys

SHARED_VENV_DIR = str(Path.home() / ".ollama-manager-venv")


class VenvManager:
    """
    Manages a single shared virtual environment for the entire project.

    Venv location : ~/.ollama-manager-venv
    Completely isolated from system Python and from Docker/Ollama.

    Usage
    -----
    python  = VenvManager.python()   # path to venv python binary
    pip     = VenvManager.pip()      # path to venv pip binary
    ok      = VenvManager.ensure()   # create venv if missing, return True on success
    ok      = VenvManager.install("turboquant")   # ensure venv + pip install
    present = VenvManager.is_installed("requests") # check if pkg is in venv
    """

    @staticmethod
    def venv_dir() -> Path:
        return Path(SHARED_VENV_DIR)

    @staticmethod
    def python() -> str:
        """Absolute path to the venv Python binary."""
        return str(VenvManager.venv_dir() / "bin" / "python")

    @staticmethod
    def pip() -> str:
        """Absolute path to the venv pip binary."""
        return str(VenvManager.venv_dir() / "bin" / "pip")

    @staticmethod
    def exists() -> bool:
        """Return True if the venv has already been created."""
        return Path(VenvManager.python()).exists()

    @staticmethod
    def ensure(verbose: bool = True) -> bool:
        """
        Create the shared venv if it does not already exist.
        Also installs/upgrades pip inside the venv.
        Returns True on success.
        """
        if VenvManager.exists():
            return True

        if verbose:
            ColorOutput.info(f"Creating shared venv at {SHARED_VENV_DIR} ...")

        # Step 1 — make sure python3-venv is available on the system
        apt_ok = subprocess.run(
            "sudo apt install -y python3-pip python3-venv python3-full",
            shell=True, timeout=120
        )
        # Non-fatal — venv module may already be present even if apt fails

        # Step 2 — create the venv
        try:
            result = subprocess.run(
                [_sys.executable, "-m", "venv", str(VenvManager.venv_dir())],
                timeout=60
            )
            if result.returncode != 0:
                if verbose:
                    ColorOutput.error("Failed to create venv.")
                return False
        except Exception as e:
            if verbose:
                ColorOutput.error(f"venv creation error: {e}")
            return False

        # Step 3 — upgrade pip inside the new venv
        try:
            subprocess.run(
                [VenvManager.pip(), "install", "--upgrade", "pip"],
                timeout=60
            )
        except Exception:
            pass  # Non-fatal

        if verbose:
            ColorOutput.success(f"Shared venv ready: {SHARED_VENV_DIR}")
        return True

    @staticmethod
    def is_installed(package: str) -> bool:
        """Return True if *package* is installed inside the shared venv."""
        if not VenvManager.exists():
            return False
        try:
            r = subprocess.run(
                [VenvManager.pip(), "show", package],
                capture_output=True, text=True, timeout=15
            )
            return r.returncode == 0
        except Exception:
            return False

    @staticmethod
    def install(package: str, verbose: bool = True) -> bool:
        """
        Ensure the shared venv exists, then pip-install *package* into it.
        Returns True on success.
        """
        if not VenvManager.ensure(verbose=verbose):
            return False
        if verbose:
            ColorOutput.info(f"Installing {package} into shared venv...")
        try:
            result = subprocess.run(
                [VenvManager.pip(), "install", package],
                timeout=180
            )
            if result.returncode != 0:
                if verbose:
                    ColorOutput.error(f"Failed to install {package}.")
                return False
        except subprocess.TimeoutExpired:
            if verbose:
                ColorOutput.error(f"Installation of {package} timed out.")
            return False
        except Exception as e:
            if verbose:
                ColorOutput.error(f"Installation error: {e}")
            return False
        if verbose:
            ColorOutput.success(f"{package} installed.")
        return True