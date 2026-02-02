"""
Docker Manager for Ollama
Handles all Docker container operations, model management, and GPU configuration
"""

import os
import sys
import json
import subprocess
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Platform-specific imports for chat interruption
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty
    import select

from utils import (
    Platform, Colors, OllamaConfig, PlatformDetector, 
    ColorOutput, CommandResult
)

class DockerManager:
    """Manage Docker operations for Ollama"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
    
    def _run_command(self, cmd: List[str]) -> CommandResult:
        """Run a command and return structured result"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return CommandResult(result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", "Command timed out")
        except Exception as e:
            return CommandResult(-1, "", str(e))
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        try:
            # Use netstat or ss to check for port usage
            plat = PlatformDetector.get_platform()
            
            if plat in [Platform.LINUX, Platform.WSL]:
                # Try ss first (modern), then netstat (legacy)
                for cmd in [['ss', '-tuln'], ['netstat', '-tuln']]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            return f':{port}' in result.stdout or f'.{port}' in result.stdout
                    except FileNotFoundError:
                        continue
            elif plat == Platform.WINDOWS:
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return f':{port}' in result.stdout
            
            return False
        except:
            return False
    
    def _get_port_user(self, port: int) -> Optional[str]:
        """Try to identify what's using the port"""
        try:
            plat = PlatformDetector.get_platform()
            
            if plat in [Platform.LINUX, Platform.WSL]:
                # Try lsof first
                try:
                    result = subprocess.run(
                        ['lsof', '-i', f':{port}', '-P', '-n'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            # Parse the output to get process info
                            return lines[1].split()[0]  # Command name
                except FileNotFoundError:
                    pass
                
                # Try ss with process info
                try:
                    result = subprocess.run(
                        ['ss', '-tlnp', f'sport = :{port}'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and 'users:' in result.stdout:
                        return result.stdout.strip()
                except FileNotFoundError:
                    pass
            
            return None
        except:
            return None
    
    def _find_ollama_containers(self) -> List[str]:
        """Find all Ollama containers (running or stopped)"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', 'ancestor=ollama/ollama', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                containers = result.stdout.strip().split('\n')
                # Exclude our own container
                return [c for c in containers if c != self.config.container_name]
            return []
        except:
            return []
    
    def is_docker_running(self) -> bool:
        """Check if Docker daemon is running"""
        result = self._run_command(['docker', 'info'])
        return result.returncode == 0
    
    def container_exists(self) -> bool:
        """Check if Ollama container exists"""
        result = self._run_command([
            'docker', 'ps', '-a',
            '--filter', f'name=^{self.config.container_name}$',
            '--format', '{{.Names}}'
        ])
        return self.config.container_name in result.stdout
    
    def container_running(self) -> bool:
        """Check if Ollama container is running"""
        result = self._run_command([
            'docker', 'ps',
            '--filter', f'name=^{self.config.container_name}$',
            '--format', '{{.Names}}'
        ])
        return self.config.container_name in result.stdout
    
    def image_exists(self) -> bool:
        """Check if Ollama image exists locally"""
        result = self._run_command([
            'docker', 'images', '-q', self.config.image_name
        ])
        return bool(result.stdout.strip())
    
    def pull_image_with_progress(self) -> bool:
        """Pull Ollama image with live progress output using Docker API"""
        ColorOutput.print("  > Ollama image not found locally", Colors.YELLOW)
        ColorOutput.print(f"  > Pulling {self.config.image_name}...", Colors.CYAN)
        ColorOutput.print("  > Downloading ~1.5GB - this may take several minutes", Colors.GRAY)
        print()
        print("─" * 60)
        
        # Try to import docker, install if needed
        try:
            import docker
        except ImportError:
            print()
            ColorOutput.warning("Docker Python library not installed")
            ColorOutput.info("This library enables better progress tracking during downloads")
            print()
            confirm = input("Install docker library? (y/n): ").strip().lower()
            
            if confirm == 'y':
                ColorOutput.info("Installing docker library...")
                try:
                    import subprocess
                    
                    # On Windows, we need pypiwin32 for named pipe support
                    packages_to_install = ['docker']
                    if os.name == 'nt':
                        packages_to_install.append('pypiwin32')
                    
                    # Build pip install command
                    pip_args = [sys.executable, '-m', 'pip', 'install'] + packages_to_install
                    # Only add --break-system-packages on Linux (not needed on Windows)
                    if os.name != 'nt':
                        pip_args.append('--break-system-packages')
                    
                    result = subprocess.run(
                        pip_args,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        ColorOutput.success("Docker library installed successfully!")
                        print()
                        ColorOutput.info("Restarting pull operation...")
                        print()
                        import docker  # Import after installation
                    else:
                        ColorOutput.warning("Could not install docker library, using fallback method")
                        print()
                        return self._pull_fallback()
                except Exception as e:
                    ColorOutput.warning(f"Installation failed: {e}")
                    print()
                    return self._pull_fallback()
            else:
                ColorOutput.info("Using fallback method without progress tracking")
                print()
                return self._pull_fallback()
        
        try:
            import docker
            
            # Create Docker client
            client = docker.APIClient(base_url='unix://var/run/docker.sock' if os.name != 'nt' else 'npipe:////./pipe/docker_engine')
            
            # Track layers and their progress
            layers = {}
            last_status = {}
            
            # Pull with streaming
            for line in client.pull(self.config.image_name, stream=True, decode=True):
                if 'status' in line:
                    status = line['status']
                    layer_id = line.get('id', '')
                    
                    # Handle different statuses
                    if 'Pulling from' in status:
                        print(f"{status}")
                    elif status == 'Pulling fs layer' and layer_id:
                        layers[layer_id] = {'status': 'Pulling', 'current': 0, 'total': 0}
                        print(f"{layer_id}: Pulling fs layer")
                    elif status in ['Downloading', 'Extracting'] and layer_id:
                        progress_detail = line.get('progressDetail', {})
                        current = progress_detail.get('current', 0)
                        total = progress_detail.get('total', 0)
                        
                        # Update layer info
                        if layer_id not in layers:
                            layers[layer_id] = {'status': status, 'current': current, 'total': total}
                        else:
                            layers[layer_id]['status'] = status
                            layers[layer_id]['current'] = current
                            layers[layer_id]['total'] = total
                        
                        # Show progress
                        if total > 0:
                            percent = (current / total) * 100
                            mb_current = current / (1024 * 1024)
                            mb_total = total / (1024 * 1024)
                            
                            # Create progress bar
                            bar_length = 30
                            filled = int(bar_length * current / total)
                            bar = '=' * filled + '>' + ' ' * (bar_length - filled - 1)
                            
                            # Only print if status changed significantly to reduce flicker
                            status_key = f"{layer_id}_{status}_{int(percent)}"
                            if status_key != last_status.get(layer_id):
                                print(f"\r{layer_id}: {status:12s} [{bar}] {percent:5.1f}% ({mb_current:.1f}/{mb_total:.1f}MB)", end='')
                                sys.stdout.flush()
                                last_status[layer_id] = status_key
                    elif status in ['Verifying Checksum', 'Download complete', 'Pull complete'] and layer_id:
                        print(f"\r{layer_id}: {status}" + " " * 50)  # Clear line
                        if layer_id in layers:
                            layers[layer_id]['status'] = status
                    elif status.startswith('Digest:') or status.startswith('Status:'):
                        print(f"\n{status}")
            
            print()
            print("─" * 60)
            print()
            ColorOutput.success("  > Image pulled successfully!")
            return True
            
        except Exception as e:
            ColorOutput.error(f"  > Error pulling image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _pull_fallback(self) -> bool:
        """Fallback method to pull image without docker-py library"""
        try:
            result = subprocess.run(
                ['docker', 'pull', self.config.image_name],
                capture_output=False,  # Show output directly
                text=True
            )
            
            print()
            print("─" * 60)
            print()
            
            if result.returncode == 0:
                ColorOutput.success("  > Image pulled successfully!")
                return True
            else:
                ColorOutput.error("  > Failed to pull image")
                return False
        except FileNotFoundError:
            ColorOutput.error("  > Docker command not found. Is Docker installed?")
            return False
        except Exception as e:
            ColorOutput.error(f"  > Error pulling image: {e}")
            return False
    
    def _check_sudo_access(self) -> bool:
        """Check if we have passwordless sudo access"""
        try:
            result = subprocess.run(['sudo', '-n', 'true'], 
                                  capture_output=True, timeout=2)
            return result.returncode == 0
        except:
            return False
    
    def _check_firewall_status(self) -> Dict:
        """Check firewall status and return diagnostics"""
        plat = PlatformDetector.get_platform()
        
        if plat in [Platform.LINUX, Platform.WSL]:
            return self._check_linux_firewall()
        elif plat == Platform.WINDOWS:
            return self._check_windows_firewall()
        
        return {'active': False, 'type': 'unknown', 'needs_config': False}
    
    def _check_linux_firewall(self) -> Dict:
        """Check Linux firewall (ufw or iptables)"""
        result = {'active': False, 'type': None, 'needs_config': False, 'rule_exists': False}
        
        # Check UFW first (most common on Ubuntu/Debian)
        try:
            ufw_status = subprocess.run(['sudo', '-n', 'ufw', 'status'], 
                                       capture_output=True, text=True, timeout=5)
            if ufw_status.returncode == 0:
                result['type'] = 'ufw'
                if 'Status: active' in ufw_status.stdout:
                    result['active'] = True
                    # Check if rule for port exists (looking for simple "11434/tcp" format)
                    if str(self.config.ollama_port) in ufw_status.stdout and 'ALLOW' in ufw_status.stdout:
                        result['rule_exists'] = True
                    else:
                        result['needs_config'] = True
                return result
        except:
            pass
        
        # Check iptables
        try:
            iptables_check = subprocess.run(['sudo', '-n', 'iptables', '-L', 'INPUT', '-n'],
                                           capture_output=True, text=True, timeout=5)
            if iptables_check.returncode == 0:
                result['type'] = 'iptables'
                result['active'] = True
                # Check for port in ACCEPT rules
                if str(self.config.ollama_port) in iptables_check.stdout and 'ACCEPT' in iptables_check.stdout:
                    result['rule_exists'] = True
                else:
                    result['needs_config'] = True
                return result
        except:
            pass
        
        return result
    
    def _check_windows_firewall(self) -> Dict:
        """Check Windows Firewall"""
        result = {'active': False, 'type': 'windows', 'needs_config': False, 'rule_exists': False}
        
        try:
            # Check if firewall is active
            fw_check = subprocess.run(
                ['powershell', '-Command', 
                 'Get-NetFirewallProfile -Profile Domain,Public,Private | Select-Object -ExpandProperty Enabled'],
                capture_output=True, text=True, timeout=5
            )
            
            if fw_check.returncode == 0 and 'True' in fw_check.stdout:
                result['active'] = True
                
                # Check for existing rule for Ollama
                rule_check = subprocess.run(
                    ['powershell', '-Command',
                     'Get-NetFirewallRule -DisplayName "Ollama*" | Select-Object -ExpandProperty DisplayName'],
                    capture_output=True, text=True, timeout=5
                )
                
                if 'Ollama' in rule_check.stdout:
                    result['rule_exists'] = True
                else:
                    result['needs_config'] = True
        except:
            pass
        
        return result
    
    def _configure_firewall_for_network_access(self) -> Tuple[bool, str]:
        """Configure firewall to allow Ollama access through Tailscale"""
        plat = PlatformDetector.get_platform()
        
        if plat in [Platform.LINUX, Platform.WSL]:
            return self._configure_linux_firewall()
        elif plat == Platform.WINDOWS:
            return self._configure_windows_firewall()
        
        return False, "Unsupported platform"
    
    def _configure_linux_firewall(self) -> Tuple[bool, str]:
        """Configure Linux firewall for Ollama network access"""
        firewall_info = self._check_linux_firewall()
        
        if not firewall_info['active']:
            return True, "No firewall active, no configuration needed"
        
        if firewall_info['rule_exists']:
            return True, "Firewall rule already exists"
        
        # Try to add rule - allow port on ALL interfaces, not just Tailscale
        if firewall_info['type'] == 'ufw':
            try:
                ColorOutput.print("  > Configuring UFW firewall for network access...", Colors.CYAN)
                result = subprocess.run(
                    ['sudo', 'ufw', 'allow', str(self.config.ollama_port) + '/tcp'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return True, "UFW firewall rule added successfully ✓"
                else:
                    return False, f"Failed to add UFW rule: {result.stderr}"
            except Exception as e:
                return False, f"Error configuring UFW: {e}"
        
        elif firewall_info['type'] == 'iptables':
            try:
                ColorOutput.print("  > Configuring iptables firewall for network access...", Colors.CYAN)
                result = subprocess.run(
                    ['sudo', 'iptables', '-I', 'INPUT', '-p', 'tcp',
                     '--dport', str(self.config.ollama_port), '-j', 'ACCEPT'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # Try to save the rule
                    subprocess.run(['sudo', 'netfilter-persistent', 'save'], 
                                 capture_output=True, timeout=5)
                    return True, "iptables firewall rule added successfully ✓"
                else:
                    return False, f"Failed to add iptables rule: {result.stderr}"
            except Exception as e:
                return False, f"Error configuring iptables: {e}"
        
        return False, "Unknown firewall type"
    
    def _configure_windows_firewall(self) -> Tuple[bool, str]:
        """Configure Windows Firewall for Ollama"""
        try:
            ColorOutput.print("  > Configuring Windows Firewall for network access...", Colors.CYAN)
            result = subprocess.run(
                ['powershell', '-Command',
                 f'New-NetFirewallRule -DisplayName "Ollama Network Access" '
                 f'-Direction Inbound -LocalPort {self.config.ollama_port} '
                 f'-Protocol TCP -Action Allow'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 or 'already exists' in result.stderr.lower():
                return True, "Windows Firewall rule added successfully ✓"
            else:
                return False, f"Failed to add Windows Firewall rule: {result.stderr}"
        except Exception as e:
            return False, f"Error configuring Windows Firewall: {e}"
    
    def _remove_firewall_rule(self) -> Tuple[bool, str]:
        """Remove Ollama firewall rule when network access is disabled"""
        plat = PlatformDetector.get_platform()
        
        if plat in [Platform.LINUX, Platform.WSL]:
            return self._remove_linux_firewall_rule()
        elif plat == Platform.WINDOWS:
            return self._remove_windows_firewall_rule()
        
        return True, "No firewall configuration to remove"
    
    def _remove_linux_firewall_rule(self) -> Tuple[bool, str]:
        """Remove Linux firewall rule for Ollama"""
        firewall_info = self._check_linux_firewall()
        
        if not firewall_info['active']:
            return True, "No firewall active"
        
        if not firewall_info['rule_exists']:
            return True, "No firewall rule to remove"
        
        # Try to remove rule
        if firewall_info['type'] == 'ufw':
            try:
                ColorOutput.print("  > Removing UFW firewall rule...", Colors.CYAN)
                # Delete the rule
                result = subprocess.run(
                    ['sudo', 'ufw', 'delete', 'allow', str(self.config.ollama_port) + '/tcp'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return True, "UFW firewall rule removed successfully ✓"
                else:
                    return False, f"Failed to remove UFW rule: {result.stderr}"
            except Exception as e:
                return False, f"Error removing UFW rule: {e}"
        
        elif firewall_info['type'] == 'iptables':
            try:
                ColorOutput.print("  > Removing iptables firewall rule...", Colors.CYAN)
                result = subprocess.run(
                    ['sudo', 'iptables', '-D', 'INPUT', '-p', 'tcp',
                     '--dport', str(self.config.ollama_port), '-j', 'ACCEPT'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # Try to save the changes
                    subprocess.run(['sudo', 'netfilter-persistent', 'save'], 
                                 capture_output=True, timeout=5)
                    return True, "iptables firewall rule removed successfully ✓"
                else:
                    return False, f"Failed to remove iptables rule: {result.stderr}"
            except Exception as e:
                return False, f"Error removing iptables rule: {e}"
        
        return False, "Unknown firewall type"
    
    def _remove_windows_firewall_rule(self) -> Tuple[bool, str]:
        """Remove Windows Firewall rule for Ollama"""
        try:
            ColorOutput.print("  > Removing Windows Firewall rule...", Colors.CYAN)
            result = subprocess.run(
                ['powershell', '-Command',
                 'Remove-NetFirewallRule -DisplayName "Ollama Network Access"'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 or 'No MSFT_NetFirewallRule objects found' in result.stderr:
                return True, "Windows Firewall rule removed successfully ✓"
            else:
                return False, f"Failed to remove Windows Firewall rule: {result.stderr}"
        except Exception as e:
            return False, f"Error removing Windows Firewall rule: {e}"
    
    def _get_port_user(self, port: int) -> Optional[str]:
        """Try to identify what's using a port"""
        try:
            # Try lsof first (most detailed)
            result = subprocess.run(
                ['lsof', '-i', f':{port}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    # Parse the output (format: COMMAND PID USER FD TYPE ...)
                    parts = lines[1].split()
                    if len(parts) >= 2:
                        return f"{parts[0]} (PID: {parts[1]})"
        except:
            pass
        
        # Try netstat with process info
        try:
            result = subprocess.run(
                ['netstat', '-tulpn'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if f':{port} ' in line:
                        # Try to extract process info
                        parts = line.split()
                        if len(parts) >= 7 and '/' in parts[-1]:
                            return parts[-1]
        except:
            pass
        
        return None
    
    def _find_ollama_containers(self) -> List[str]:
        """Find any running ollama containers"""
        try:
            result = self._run_command([
                'docker', 'ps', '--format', '{{.Names}}',
                '--filter', 'name=ollama'
            ])
            if result.returncode == 0:
                containers = [name.strip() for name in result.stdout.split('\n') if name.strip()]
                # Exclude our own container
                return [c for c in containers if c != self.config.container_name]
        except:
            pass
        return []
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        try:
            result = self._run_command([
                'docker', 'ps', '--format', '{{.Ports}}'
            ])
            if result.returncode == 0:
                # Check if our port appears in any container's port mappings
                return f':{port}->' in result.stdout or f':{port}/' in result.stdout
        except:
            pass
        
        # Also check system ports using netstat/ss
        try:
            # Try ss first (modern Linux)
            result = subprocess.run(
                ['ss', '-tuln'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and f':{port} ' in result.stdout:
                return True
        except:
            try:
                # Fallback to netstat
                result = subprocess.run(
                    ['netstat', '-tuln'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and f':{port} ' in result.stdout:
                    return True
            except:
                pass
        
        return False
    
    def _find_available_port(self, start_port: int = 11434, max_attempts: int = 50) -> Optional[int]:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if not self._is_port_in_use(port):
                return port
        return None
    
    def _handle_port_conflict(self) -> bool:
        """Handle port conflict - let user choose a new port or find one automatically"""
        ColorOutput.error(f"Port {self.config.ollama_port} is already in use!")
        print()
        ColorOutput.print("Options:", Colors.CYAN, bold=True)
        print("  [A] Automatically find an available port")
        print("  [M] Manually enter a different port")
        print("  [C] Cancel and return to menu")
        print()
        
        choice = input("Select option (A/M/C): ").strip().upper()
        
        if choice == 'A':
            # Find available port automatically
            ColorOutput.info("Searching for available port...")
            new_port = self._find_available_port(self.config.ollama_port + 1)
            
            if new_port:
                ColorOutput.success(f"Found available port: {new_port}")
                print()
                confirm = input(f"Use port {new_port}? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.config.ollama_port = new_port
                    self._save_config()
                    ColorOutput.success(f"Port updated to {new_port}")
                    return True
                else:
                    ColorOutput.info("Port change cancelled")
                    return False
            else:
                ColorOutput.error("Could not find an available port in range")
                return False
                
        elif choice == 'M':
            # Manual port entry
            while True:
                try:
                    print()
                    new_port_str = input(f"Enter port number (1024-65535, current: {self.config.ollama_port}): ").strip()
                    new_port = int(new_port_str)
                    
                    if new_port < 1024 or new_port > 65535:
                        ColorOutput.error("Port must be between 1024 and 65535")
                        continue
                    
                    if self._is_port_in_use(new_port):
                        ColorOutput.error(f"Port {new_port} is also in use!")
                        retry = input("Try another port? (y/n): ").strip().lower()
                        if retry != 'y':
                            return False
                        continue
                    
                    # Port is valid and available
                    self.config.ollama_port = new_port
                    self._save_config()
                    ColorOutput.success(f"Port updated to {new_port}")
                    return True
                    
                except ValueError:
                    ColorOutput.error("Invalid port number")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        return False
                except KeyboardInterrupt:
                    print()
                    ColorOutput.info("Port change cancelled")
                    return False
        else:
            ColorOutput.info("Operation cancelled")
            return False
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'ContainerName': self.config.container_name,
                'OllamaPort': self.config.ollama_port,
                'UseGPU': self.config.use_gpu,
                'NetworkAccess': self.config.network_access
            }
            
            with open(self.config.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            ColorOutput.warning(f"Could not save config: {e}")
    
    def start_container(self) -> bool:
        """Start or create Ollama container"""
        container_exists = self.container_exists()
        container_running = self.container_running()
        
        # Check if image exists first
        if not self.image_exists():
            ColorOutput.error("Ollama image not installed!")
            ColorOutput.info("Please install Ollama first using option [I]")
            return False
        
        # Ensure models directory exists
        success, models_dir = PlatformDetector.ensure_models_directory()
        if not success:
            ColorOutput.error(f"Failed to create models directory: {models_dir}")
            return False
        
        # Check if port is in use (but only for new containers)
        if not container_exists:
            if self._is_port_in_use(self.config.ollama_port):
                if not self._handle_port_conflict():
                    return False
                # Port was updated, continue with new port
        
        if container_running:
            ColorOutput.info("Restarting Ollama container...")
            # Stop first
            self._run_command(['docker', 'stop', self.config.container_name])
            # Then start again
            result = self._run_command(['docker', 'start', self.config.container_name])
            if result.returncode == 0:
                print()
                ColorOutput.success("Ollama container restarted successfully!")
                return True
            else:
                ColorOutput.error(f"Failed to restart container: {result.stderr}")
                return False
        
        if container_exists:
            ColorOutput.info("Starting Ollama container...")
        else:
            ColorOutput.info("Creating and starting Ollama container...")
            ColorOutput.print(f"  > Container name: {self.config.container_name}", Colors.CYAN)
            
            # Check if port is already in use before creating container
            if self._is_port_in_use(self.config.ollama_port):
                print()
                ColorOutput.error(f"Port {self.config.ollama_port} is already in use!")
                print()
                
                # Try to identify what's using the port
                port_user = self._get_port_user(self.config.ollama_port)
                if port_user:
                    ColorOutput.print(f"Port is being used by: {port_user}", Colors.YELLOW)
                    print()
                
                ColorOutput.print("Solutions:", Colors.CYAN, bold=True)
                print("  1. Stop the process using this port")
                print("  2. Change Ollama port in Settings [S]")
                print()
                
                # Offer to check for existing Ollama containers
                other_containers = self._find_ollama_containers()
                if other_containers:
                    ColorOutput.warning("Found other Ollama containers:")
                    for container in other_containers:
                        print(f"    • {container}")
                    print()
                    ColorOutput.info("You may need to stop these containers first:")
                    print(f"    docker stop {' '.join(other_containers)}")
                
                return False
        
        # Show GPU mode status
        if self.config.use_gpu:
            ColorOutput.print("  > GPU Mode: Enabled", Colors.GREEN)
        else:
            ColorOutput.print("  > GPU Mode: Disabled (CPU only)", Colors.CYAN)
        
        try:
            if container_exists:
                ColorOutput.print("  > Starting existing container...", Colors.CYAN)
                result = self._run_command(['docker', 'start', self.config.container_name])
                if result.returncode == 0:
                    print(f"  > Container ID: {result.stdout.strip()}")
                else:
                    error_msg = result.stderr.lower()
                    
                    # Check if it's a port binding error
                    if 'port' in error_msg and ('in use' in error_msg or 'already allocated' in error_msg or 'bind' in error_msg):
                        print()
                        ColorOutput.error("Failed to start - port conflict detected")
                        print()
                        
                        # Get info about what's using the port
                        port_user = self._get_port_user(self.config.ollama_port)
                        if port_user:
                            ColorOutput.print(f"Port {self.config.ollama_port} is being used by: {port_user}", Colors.YELLOW)
                        else:
                            ColorOutput.print(f"Port {self.config.ollama_port} is already in use", Colors.YELLOW)
                        print()
                        
                        # Check for other ollama containers
                        other_containers = self._find_ollama_containers()
                        if other_containers:
                            ColorOutput.warning("Found other Ollama containers:")
                            for container in other_containers:
                                print(f"    • {container}")
                            print()
                            ColorOutput.info("Stop them with: docker stop " + " ".join(other_containers))
                            print()
                        
                        ColorOutput.print("Solutions:", Colors.CYAN, bold=True)
                        print("  [1] Stop the process using this port")
                        print("  [2] Change the port in Settings [S]")
                        print("  [3] Remove and recreate container with new port [R]")
                        print()
                        
                        return False
                    else:
                        ColorOutput.error(f"Failed to start container: {result.stderr}")
                        return False
            else:
                ColorOutput.print("  > Creating container...", Colors.CYAN)
                ColorOutput.print(f"  > Configuring port {self.config.ollama_port}...", Colors.CYAN)
                ColorOutput.print(f"  > Models directory: {models_dir}", Colors.CYAN)
                
                # Build docker run command - let Docker use system DNS
                cmd = ['docker', 'run', '-d']
                
                if self.config.use_gpu:
                    ColorOutput.print("  > Configuring GPU acceleration...", Colors.CYAN)
                    cmd.extend(['--gpus', 'all'])
                
                # Configure port binding based on network access setting
                # WSL has issues with 127.0.0.1 binding format, use plain port mapping for localhost
                plat = PlatformDetector.get_platform()
                
                if self.config.network_access:
                    port_binding = f'0.0.0.0:{self.config.ollama_port}:11434'
                    ColorOutput.print("  > Network access: ENABLED (accessible from other computers)", Colors.YELLOW)
                    
                    # Check and configure firewall for network access
                    firewall_status = self._check_firewall_status()
                    
                    if firewall_status['active'] and firewall_status['needs_config']:
                        print()
                        ColorOutput.print("  ⚠ Firewall Detected - Configuration Needed", Colors.YELLOW, bold=True)
                        ColorOutput.print(f"  > Firewall type: {firewall_status['type']}", Colors.CYAN)
                        ColorOutput.print("  > For Tailscale/network access to work, firewall must allow port", Colors.GRAY)
                        print()
                        
                        if self._check_sudo_access():
                            # We have sudo, try to configure automatically
                            ColorOutput.print("  Attempting automatic firewall configuration...", Colors.CYAN)
                            success, message = self._configure_firewall_for_network_access()
                            print()
                            if success:
                                ColorOutput.success(f"  {message}")
                            else:
                                ColorOutput.warning(f"  {message}")
                                print()
                                ColorOutput.print("  📋 Manual firewall configuration needed:", Colors.YELLOW, bold=True)
                                if firewall_status['type'] == 'ufw':
                                    print(f"    sudo ufw allow {self.config.ollama_port}/tcp")
                                elif firewall_status['type'] == 'iptables':
                                    print(f"    sudo iptables -I INPUT -p tcp --dport {self.config.ollama_port} -j ACCEPT")
                        else:
                            # No sudo access - provide instructions
                            ColorOutput.print("  📋 Manual firewall configuration required:", Colors.YELLOW, bold=True)
                            if firewall_status['type'] == 'ufw':
                                print(f"    sudo ufw allow {self.config.ollama_port}/tcp")
                            elif firewall_status['type'] == 'iptables':
                                print(f"    sudo iptables -I INPUT -p tcp --dport {self.config.ollama_port} -j ACCEPT")
                        print()
                    elif firewall_status['active'] and firewall_status['rule_exists']:
                        ColorOutput.print("  > Firewall: Already configured for network access ✓", Colors.GREEN)
                else:
                    # For localhost-only access, use different binding based on platform
                    # WSL has issues with 127.0.0.1:port:port format, use simple port mapping
                    if plat == Platform.WSL:
                        port_binding = f'{self.config.ollama_port}:11434'
                    else:
                        port_binding = f'127.0.0.1:{self.config.ollama_port}:11434'
                    ColorOutput.print("  > Network access: DISABLED (localhost only)", Colors.GRAY)
                
                cmd.extend([
                    '--name', self.config.container_name,
                    '-p', port_binding,
                    '-v', f'{models_dir}:/root/.ollama',
                    self.config.image_name
                ])
                
                result = self._run_command(cmd)
                
                if result.returncode == 0:
                    print(f"  > Container ID: {result.stdout.strip()}")
                else:
                    error_msg = result.stderr.lower()
                    
                    # Check if it's a port binding error
                    if 'port' in error_msg and ('in use' in error_msg or 'already allocated' in error_msg or 'bind' in error_msg):
                        print()
                        ColorOutput.error("Port binding failed - port may be in use")
                        print()
                        
                        # Offer to retry with different port
                        if self._handle_port_conflict():
                            ColorOutput.info("Retrying with new port...")
                            return self.start_container()  # Recursive retry with new port
                        return False
                    
                    ColorOutput.error(f"Failed to create container: {result.stderr}")
                    if self.config.use_gpu:
                        ColorOutput.warning("Hint: If GPU mode fails, try switching to CPU mode in settings")
                    return False
            
            print()
            ColorOutput.success("Ollama container started successfully!")
            ColorOutput.print(f"  Models are stored in: {models_dir}", Colors.CYAN)
            if self.config.use_gpu:
                ColorOutput.print("  GPU acceleration is active", Colors.GREEN)
            
            # Show next steps for newcomers
            if not container_exists:
                print()
                ColorOutput.print("Next steps:", Colors.CYAN, bold=True)
                ColorOutput.print("  • Press [1] to install a model (e.g., llama3.2, mistral)", Colors.WHITE)
                ColorOutput.print("  • Press [7] to chat with your installed models", Colors.WHITE)
            
            return True
            
        except Exception as e:
            print()
            ColorOutput.error(f"Failed to start Ollama container: {e}")
            if self.config.use_gpu:
                ColorOutput.warning("Hint: If GPU mode fails, try switching to CPU mode in settings")
            return False
    
    def stop_container(self) -> bool:
        """Stop Ollama container"""
        ColorOutput.info("Stopping Ollama container...")
        result = self._run_command(['docker', 'stop', self.config.container_name])
        
        if result.returncode == 0:
            print()
            ColorOutput.success("Ollama container stopped!")
            return True
        else:
            print()
            ColorOutput.error("Failed to stop Ollama container")
            return False
    
    def remove_container(self) -> bool:
        """Remove Ollama container"""
        result = self._run_command(['docker', 'rm', self.config.container_name])
        return result.returncode == 0
    
    def get_container_details(self) -> Optional[Dict]:
        """Get container details"""
        if not self.container_exists():
            return None
        
        try:
            image = self._run_command([
                'docker', 'inspect', self.config.container_name,
                '--format', '{{.Config.Image}}'
            ]).stdout.strip()
            
            created = self._run_command([
                'docker', 'inspect', self.config.container_name,
                '--format', '{{.Created}}'
            ]).stdout.strip()
            
            # Check GPU
            gpu_devices = self._run_command([
                'docker', 'inspect', self.config.container_name,
                '--format', '{{.HostConfig.DeviceRequests}}'
            ]).stdout.strip()
            
            has_gpu = gpu_devices and gpu_devices != '[]'
            
            return {
                'image': image,
                'created': created,
                'name': self.config.container_name,
                'port': self.config.ollama_port,
                'gpu_enabled': has_gpu
            }
        except:
            return None
    
    def list_models(self) -> List[str]:
        """List installed Ollama models"""
        if not self.container_running():
            ColorOutput.warning("Container is not running")
            return []
        
        result = self._run_command([
            'docker', 'exec', self.config.container_name,
            'ollama', 'list'
        ])
        
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull/install an Ollama model with progress"""
        if not self.container_running():
            ColorOutput.error("Container is not running!")
            return False
        
        # Get and display the models directory
        models_dir = PlatformDetector.get_models_directory()
        
        ColorOutput.info(f"Installing model: {model_name}")
        ColorOutput.print(f"  Storage location: {models_dir}", Colors.CYAN)
        ColorOutput.print("  This may take several minutes depending on the model size...", Colors.GRAY)
        print()
        
        try:
            # Use subprocess.call to show live progress
            returncode = subprocess.call([
                'docker', 'exec', '-it', self.config.container_name,
                'ollama', 'pull', model_name
            ])
            
            print()
            
            if returncode == 0:
                ColorOutput.success(f"Model '{model_name}' installed successfully!")
                ColorOutput.print(f"  Stored in: {models_dir}", Colors.CYAN)
                return True
            else:
                ColorOutput.error(f"Failed to install model '{model_name}'")
                return False
                
        except Exception as e:
            print()
            ColorOutput.error(f"Error installing model: {e}")
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove an Ollama model"""
        if not self.container_running():
            ColorOutput.error("Container is not running!")
            return False
        
        ColorOutput.info(f"Removing model: {model_name}")
        result = self._run_command([
            'docker', 'exec', self.config.container_name,
            'ollama', 'rm', model_name
        ])
        
        if result.returncode == 0:
            print()
            ColorOutput.success(f"Model '{model_name}' removed successfully!")
            return True
        else:
            print()
            ColorOutput.error(f"Failed to remove model '{model_name}'")
            return False
    
    def chat_with_model(self, model_name: str):
        """Start interactive chat with a model (with interrupt support)"""
        if not self.container_running():
            ColorOutput.error("Container is not running!")
            return
        
        ColorOutput.header(f"CHAT WITH {model_name.upper()}")
        ColorOutput.print("Type your messages below.", Colors.GRAY)
        ColorOutput.print("Thinking processes appear in gray, answers in white.", Colors.CYAN)
        ColorOutput.print("Press Ctrl+C during response to interrupt and ask a new question.", Colors.YELLOW)
        ColorOutput.print("Type '/bye' to exit chat.", Colors.GRAY)
        print()
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                ColorOutput.print("You: ", Colors.CYAN, bold=True, end='')
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/bye', '/exit', '/quit']:
                    ColorOutput.info("Chat ended")
                    break
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Show assistant label
                ColorOutput.print(f"\n{model_name}: ", Colors.GREEN, bold=True)
                
                # Stream response with interrupt support
                try:
                    response_text = self._stream_chat_response(model_name, conversation_history)
                    if response_text:
                        conversation_history.append({"role": "assistant", "content": response_text})
                    else:
                        # Response was interrupted, remove last user message
                        conversation_history.pop()
                        ColorOutput.print("\n[Response interrupted]", Colors.YELLOW)
                    print()
                except KeyboardInterrupt:
                    # User interrupted during response
                    conversation_history.pop()  # Remove last user message
                    ColorOutput.print("\n\n[Response interrupted - ask your next question]", Colors.YELLOW)
                    print()
                
            except KeyboardInterrupt:
                print("\n")
                ColorOutput.info("Chat ended")
                break
            except Exception as e:
                ColorOutput.error(f"Chat error: {e}")
                break
    
    def _stream_chat_response(self, model_name: str, conversation_history: List[Dict]) -> Optional[str]:
        """
        Stream chat response from Ollama API with interrupt support.
        Returns the complete response text, or None if interrupted.
        Shows all streaming content including reasoning/thinking processes.
        """
        try:
            import requests
        except ImportError:
            ColorOutput.error("\nError: 'requests' library not found!")
            ColorOutput.print("Install it with: pip install requests", Colors.YELLOW)
            return None
        
        # Prepare the API request
        url = f"http://localhost:{self.config.ollama_port}/api/chat"
        payload = {
            "model": model_name,
            "messages": conversation_history,
            "stream": True
        }
        
        response_text = ""
        
        try:
            # Make streaming request
            response = requests.post(url, json=payload, stream=True, timeout=300)
            response.raise_for_status()
            
            # Set stdin to non-blocking mode on Unix
            if os.name != 'nt':
                import termios
                import tty
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            
            try:
                # Track if we're transitioning from thinking to answer
                last_was_thinking = False
                
                for line in response.iter_lines():
                    # Check for interrupt (Ctrl+C or any key press)
                    if self._check_interrupt():
                        raise KeyboardInterrupt()
                    
                    if line:
                        try:
                            chunk = json.loads(line)
                            
                            # Stream all content as it arrives
                            if 'message' in chunk:
                                message = chunk['message']
                                
                                # Check if we have thinking/reasoning content
                                has_thinking = ('thinking' in message and message['thinking']) or \
                                             ('reasoning' in message and message['reasoning'])
                                has_content = 'content' in message and message.get('content', '')
                                
                                # Capture thinking/reasoning fields if present
                                # Print these in gray to distinguish from main answer
                                if 'thinking' in message and message['thinking']:
                                    thinking = message['thinking']
                                    # Print thinking in gray
                                    print(f"{Colors.GRAY}{thinking}{Colors.RESET}", end='', flush=True)
                                    response_text += thinking
                                    last_was_thinking = True
                                
                                if 'reasoning' in message and message['reasoning']:
                                    reasoning = message['reasoning']
                                    # Print reasoning in gray
                                    print(f"{Colors.GRAY}{reasoning}{Colors.RESET}", end='', flush=True)
                                    response_text += reasoning
                                    last_was_thinking = True
                                
                                # Get main content (print in normal white)
                                content = message.get('content', '')
                                if content:
                                    # Add newline if transitioning from thinking to answer
                                    if last_was_thinking:
                                        print("\n", end='', flush=True)
                                        last_was_thinking = False
                                    
                                    print(content, end='', flush=True)
                                    response_text += content
                            
                            # Check if done
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            finally:
                # Restore stdin settings on Unix
                if os.name != 'nt':
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            ColorOutput.error(f"\nAPI Error: {e}")
            ColorOutput.print("Make sure the Ollama container is running and the model is installed", Colors.GRAY)
            return None
        except KeyboardInterrupt:
            # Restore stdin on interrupt
            if os.name != 'nt':
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except:
                    pass
            return None
    
    def _check_interrupt(self) -> bool:
        """Check if user pressed a key (non-blocking)"""
        if os.name == 'nt':  # Windows
            try:
                import msvcrt
                if msvcrt.kbhit():
                    # Read and discard the character
                    ch = msvcrt.getch()
                    # Check if it's Ctrl+C (ASCII 3)
                    if ch == b'\x03':
                        return True
                    return False
            except:
                return False
        else:  # Unix/Linux
            return select.select([sys.stdin], [], [], 0)[0]
    
    def get_loaded_models(self) -> List[Dict]:
        """Get list of currently loaded models in memory"""
        if not self.container_running():
            return []
        
        try:
            cmd = [
                'docker', 'exec', self.config.container_name,
                'ollama', 'ps'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) <= 1:  # Only header or empty
                    return []
                
                loaded_models = []
                # Skip header line
                for line in lines[1:]:
                    if line.strip():
                        # Parse the line (NAME, ID, SIZE, PROCESSOR, UNTIL)
                        parts = line.split()
                        if len(parts) >= 1:
                            loaded_models.append({
                                'name': parts[0],
                                'full_line': line
                            })
                
                return loaded_models
            return []
        except Exception as e:
            ColorOutput.warning(f"Could not get loaded models: {e}")
            return []
    
    def load_model(self, model_name: str) -> bool:
        """Load a model into memory"""
        if not self.container_running():
            ColorOutput.error("Container is not running")
            return False
        
        try:
            ColorOutput.info(f"Loading model into memory: {model_name}")
            ColorOutput.print("This may take a moment depending on model size...", Colors.GRAY)
            
            # Load the model by sending an empty prompt with keep_alive
            cmd = [
                'docker', 'exec', self.config.container_name,
                'curl', '-X', 'POST', 'http://localhost:11434/api/generate',
                '-d', f'{{"model": "{model_name}", "prompt": "", "keep_alive": "5m"}}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                ColorOutput.success(f"Model {model_name} has been loaded into memory")
                ColorOutput.print("  Model will stay loaded for 5 minutes of inactivity", Colors.GRAY)
                return True
            else:
                ColorOutput.error(f"Failed to load model: {result.stderr}")
                return False
        except Exception as e:
            ColorOutput.error(f"Error loading model: {e}")
            return False
    
    def unload_model(self, model_name: str, force: bool = False) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_name: Name of the model to unload
            force: If True, tries multiple methods to unload the model
        
        Returns:
            True if successful, False otherwise
        """
        if not self.container_running():
            ColorOutput.error("Container is not running")
            return False
        
        try:
            import requests
        except ImportError:
            ColorOutput.error("'requests' library not found! Install with: pip install requests")
            return False
        
        ColorOutput.info(f"Unloading model: {model_name}")
        
        # Method 1: Use the Ollama API generate endpoint with keep_alive=0
        try:
            url = f"http://localhost:{self.config.ollama_port}/api/generate"
            payload = {
                "model": model_name,
                "keep_alive": 0
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                ColorOutput.success(f"✓ Model '{model_name}' has been unloaded from memory")
                return True
            elif response.status_code == 404:
                ColorOutput.warning(f"Model '{model_name}' was not loaded in memory")
                return True  # Not loaded = mission accomplished
            else:
                if not force:
                    ColorOutput.error(f"Failed to unload (HTTP {response.status_code})")
                    ColorOutput.print(f"Response: {response.text[:200]}", Colors.GRAY)
                    return False
                # If force mode, continue to try alternative methods
                ColorOutput.warning(f"Method 1 failed (HTTP {response.status_code}), trying alternative...")
        
        except requests.exceptions.ConnectionError:
            ColorOutput.error(f"Cannot connect to Ollama API on port {self.config.ollama_port}")
            ColorOutput.print("Make sure the Ollama container is running", Colors.GRAY)
            return False
        except requests.exceptions.Timeout:
            if not force:
                ColorOutput.error("Request timed out after 30 seconds")
                return False
            ColorOutput.warning("Method 1 timed out, trying alternative...")
        except Exception as e:
            if not force:
                ColorOutput.error(f"Error unloading model: {e}")
                return False
            ColorOutput.warning(f"Method 1 error ({e}), trying alternative...")
        
        # Method 2 (Force mode): Try using the /api/show endpoint with keep_alive=0
        if force:
            try:
                ColorOutput.info("Trying force unload method 2...")
                url = f"http://localhost:{self.config.ollama_port}/api/show"
                payload = {
                    "model": model_name,
                    "keep_alive": 0
                }
                
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code in [200, 404]:
                    ColorOutput.success(f"✓ Model '{model_name}' unloaded (method 2)")
                    return True
                else:
                    ColorOutput.warning(f"Method 2 failed (HTTP {response.status_code})")
            except Exception as e:
                ColorOutput.warning(f"Method 2 error: {e}")
        
        # Method 3 (Force mode): Restart the container to clear all models
        if force:
            ColorOutput.warning("All unload methods failed")
            print()
            ColorOutput.print("Forceful options:", Colors.YELLOW, bold=True)
            print("  1. Restart the Ollama container (clears ALL models from memory)")
            print("  2. Cancel and try manual unload")
            print()
            
            choice = input("Select option (1/2): ").strip()
            
            if choice == '1':
                ColorOutput.info("Restarting container to clear all models from memory...")
                if self.stop_container() and self.start_container():
                    ColorOutput.success("Container restarted - all models unloaded")
                    return True
                else:
                    ColorOutput.error("Failed to restart container")
                    return False
            else:
                ColorOutput.info("Operation cancelled")
                return False
        
        return False
    
    def complete_removal(self) -> bool:
        """Complete removal of Ollama (container, image, and volume)"""
        ColorOutput.header("COMPLETE REMOVAL")
        ColorOutput.warning("This will remove:")
        print("  • Ollama Docker container")
        print("  • Ollama Docker image")
        print()
        
        models_dir = PlatformDetector.get_models_directory()
        ColorOutput.print("What will NOT be deleted:", Colors.CYAN, bold=True)
        print(f"  • Your downloaded models in: {models_dir}")
        print("  • Configuration file: ollama-config.json")
        print()
        ColorOutput.print("Note:", Colors.YELLOW)
        print("  Your models stay on disk and will be reused if you reinstall Ollama.")
        print("  To free up disk space, manually delete the models directory after removal.")
        print()
        
        confirm = input("Are you absolutely sure? (yes/no): ").strip().lower()
        if confirm != 'yes':
            ColorOutput.info("Removal cancelled")
            return False
        
        success = True
        
        # Stop container if running
        if self.container_running():
            ColorOutput.info("Stopping container...")
            if not self.stop_container():
                success = False
        
        # Remove container
        if self.container_exists():
            ColorOutput.info("Removing container...")
            if self.remove_container():
                ColorOutput.success("Container removed")
            else:
                ColorOutput.error("Failed to remove container")
                success = False
        
        # Remove image
        if self.image_exists():
            ColorOutput.info("Removing Docker image...")
            result = self._run_command(['docker', 'rmi', self.config.image_name])
            if result.returncode == 0:
                ColorOutput.success("Docker image removed")
            else:
                ColorOutput.error("Failed to remove Docker image")
                success = False
        
        # Clean up firewall rule if it exists
        firewall_status = self._check_firewall_status()
        if firewall_status['active'] and firewall_status['rule_exists']:
            print()
            ColorOutput.print("Firewall cleanup:", Colors.CYAN, bold=True)
            ColorOutput.print("  A firewall rule exists for Ollama network access", Colors.GRAY)
            cleanup = input("Remove firewall rule? (y/n): ").strip().lower()
            
            if cleanup == 'y':
                if self._check_sudo_access():
                    fw_success, fw_message = self._remove_firewall_rule()
                    print()
                    if fw_success:
                        ColorOutput.success(f"  {fw_message}")
                    else:
                        ColorOutput.warning(f"  {fw_message}")
                else:
                    print()
                    ColorOutput.print("  Manual cleanup required:", Colors.YELLOW)
                    if firewall_status['type'] == 'ufw':
                        print(f"    sudo ufw delete allow {self.config.ollama_port}/tcp")
                    elif firewall_status['type'] == 'iptables':
                        print(f"    sudo iptables -D INPUT -p tcp --dport {self.config.ollama_port} -j ACCEPT")

        
        if success:
            print()
            ColorOutput.success("Ollama Docker components removed successfully!")
            print()
            ColorOutput.print("Your models are still in:", Colors.CYAN, bold=True)
            print(f"  {models_dir}")
            print()
            ColorOutput.print("If you reinstall Ollama, your models will still be there!", Colors.GREEN)
        
        return success
    
    def full_uninstall(self) -> bool:
        """Complete uninstall - removes EVERYTHING including models directory"""
        ColorOutput.header("FULL UNINSTALL - DELETE EVERYTHING")
        
        models_dir = PlatformDetector.get_models_directory()
        
        ColorOutput.print("⚠️  WARNING - THIS WILL DELETE EVERYTHING! ⚠️", Colors.RED, bold=True)
        print()
        ColorOutput.warning("This will permanently remove:")
        print("  • Ollama Docker container")
        print("  • Ollama Docker image")
        print(f"  • ALL downloaded models in: {models_dir}")
        print("  • Configuration file: ollama-config.json")
        print()
        
        ColorOutput.print("This action CANNOT be undone!", Colors.RED, bold=True)
        print()
        
        # Check if models directory exists and has content
        models_path = Path(models_dir)
        if models_path.exists():
            try:
                # Try to estimate size
                total_size = sum(f.stat().st_size for f in models_path.rglob('*') if f.is_file())
                size_gb = total_size / (1024**3)
                if size_gb > 0.1:
                    ColorOutput.warning(f"Models directory contains approximately {size_gb:.2f} GB of data")
                    print()
            except:
                pass
        
        confirm1 = input("Type 'DELETE EVERYTHING' to confirm (or anything else to cancel): ").strip()
        if confirm1 != 'DELETE EVERYTHING':
            ColorOutput.info("Full uninstall cancelled")
            return False
        
        print()
        ColorOutput.warning("Are you absolutely certain? This will delete ALL your models!")
        confirm2 = input("Type 'YES I AM SURE' to proceed: ").strip()
        if confirm2 != 'YES I AM SURE':
            ColorOutput.info("Full uninstall cancelled")
            return False
        
        print()
        success = True
        
        # Stop container if running
        if self.container_running():
            ColorOutput.info("Stopping container...")
            if not self.stop_container():
                success = False
        
        # Remove container
        if self.container_exists():
            ColorOutput.info("Removing container...")
            if self.remove_container():
                ColorOutput.success("Container removed")
            else:
                ColorOutput.error("Failed to remove container")
                success = False
        
        # Remove image
        if self.image_exists():
            ColorOutput.info("Removing Docker image...")
            result = self._run_command(['docker', 'rmi', self.config.image_name])
            if result.returncode == 0:
                ColorOutput.success("Docker image removed")
            else:
                ColorOutput.error("Failed to remove Docker image")
                success = False
        
        # Delete models directory
        if models_path.exists():
            ColorOutput.info(f"Deleting models directory: {models_dir}")
            try:
                import shutil
                shutil.rmtree(models_dir)
                ColorOutput.success("Models directory deleted")
            except Exception as e:
                ColorOutput.error(f"Failed to delete models directory: {e}")
                ColorOutput.print(f"  You may need to manually delete: {models_dir}", Colors.YELLOW)
                success = False
        
        # Delete config file
        config_file = Path(self.config.config_file)
        if config_file.exists():
            ColorOutput.info("Deleting configuration file...")
            try:
                config_file.unlink()
                ColorOutput.success("Configuration file deleted")
            except Exception as e:
                ColorOutput.error(f"Failed to delete config file: {e}")
                success = False
        
        # Clean up firewall rule if it exists
        firewall_status = self._check_firewall_status()
        if firewall_status['active'] and firewall_status['rule_exists']:
            print()
            ColorOutput.print("Firewall cleanup:", Colors.CYAN, bold=True)
            if self._check_sudo_access():
                fw_success, fw_message = self._remove_firewall_rule()
                if fw_success:
                    ColorOutput.success(f"  {fw_message}")
                else:
                    ColorOutput.warning(f"  {fw_message}")
            else:
                ColorOutput.print("  Manual cleanup required:", Colors.YELLOW)
                if firewall_status['type'] == 'ufw':
                    print(f"    sudo ufw delete allow {self.config.ollama_port}/tcp")
                elif firewall_status['type'] == 'iptables':
                    print(f"    sudo iptables -D INPUT -p tcp --dport {self.config.ollama_port} -j ACCEPT")
        
        if success:
            print()
            ColorOutput.success("Full uninstall completed!")
            ColorOutput.print("  Everything has been removed from your system", Colors.GREEN)
            print()
            ColorOutput.print("You can reinstall Ollama anytime using option [I]", Colors.CYAN)
        
        return success