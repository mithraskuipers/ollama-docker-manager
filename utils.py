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
            'issues': [],
            'recommendations': []
        }
        
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
                result['recommendations'].append('Install NVIDIA drivers:')
                result['recommendations'].append('  Ubuntu/Debian: sudo apt install nvidia-driver-XXX')
                result['recommendations'].append('  Or use: ubuntu-drivers devices && sudo ubuntu-drivers autoinstall')
                return result
        except FileNotFoundError:
            result['issues'].append('nvidia-smi not found')
            result['recommendations'].append('Install NVIDIA drivers first')
            return result
        
        # Check 2: nvidia-docker / nvidia-container-toolkit
        try:
            docker_check = subprocess.run(
                ['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0-base', 'nvidia-smi'],
                capture_output=True, timeout=10
            )
            
            if docker_check.returncode == 0:
                result['docker_support'] = True
                result['available'] = True
            else:
                result['issues'].append('Docker GPU support not configured')
                result['recommendations'].append('Install NVIDIA Container Toolkit:')
                result['recommendations'].append('  1. Add repository:')
                result['recommendations'].append('     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)')
                result['recommendations'].append('     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -')
                result['recommendations'].append('     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list')
                result['recommendations'].append('  2. Install:')
                result['recommendations'].append('     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit')
                result['recommendations'].append('  3. Restart Docker:')
                result['recommendations'].append('     sudo systemctl restart docker')
        except Exception:
            result['issues'].append('Could not test Docker GPU support')
            result['recommendations'].append('Ensure nvidia-container-toolkit is installed')
        
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
