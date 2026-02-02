<#
.SYNOPSIS
    Ollama Docker Manager - Windows Launcher
.DESCRIPTION
    Cross-platform launcher for Ollama Docker Manager
    Automatically detects Python and runs the manager
#>

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "           OLLAMA DOCKER MANAGER - Cross Platform" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = $null
$pythonVersions = @("python", "python3", "py")

foreach ($cmd in $pythonVersions) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "[OK] Found Python: $version" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.7 or later from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Checking Python dependencies..." -ForegroundColor Cyan

# Check for requests library
try {
    & $pythonCmd -c "import requests" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Python 'requests' library found" -ForegroundColor Green
    } else {
        throw
    }
} catch {
    Write-Host "[WARN] Python 'requests' library not found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Installing 'requests' library..." -ForegroundColor Cyan
    
    try {
        & $pythonCmd -m pip install requests --user 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Successfully installed 'requests'" -ForegroundColor Green
        } else {
            Write-Host "Note: The chat feature requires the 'requests' library" -ForegroundColor Yellow
            Write-Host "Install manually with: $pythonCmd -m pip install requests" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "Note: The chat feature requires the 'requests' library" -ForegroundColor Yellow
        Write-Host "Install manually with: $pythonCmd -m pip install requests" -ForegroundColor Cyan
    }
    Write-Host ""
}

Write-Host ""
Write-Host "Checking Docker status..." -ForegroundColor Cyan

try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        # Docker command exists, now check if daemon is running (with timeout)
        Write-Host "Testing Docker daemon..." -ForegroundColor Gray
        
        # Use Start-Process with timeout for docker info
        $job = Start-Job -ScriptBlock { docker info 2>&1 }
        $completed = Wait-Job -Job $job -Timeout 5
        
        if ($completed) {
            $dockerInfo = Receive-Job -Job $job
            Remove-Job -Job $job -Force
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Found Docker: $dockerVersion" -ForegroundColor Green
                Write-Host "[OK] Docker daemon is running" -ForegroundColor Green
            } else {
                Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
                Write-Host ""
                Write-Host "[ERROR] Docker is installed but NOT running!" -ForegroundColor Red
                Write-Host ""
                Write-Host "Docker Desktop is not currently running." -ForegroundColor Yellow
                Write-Host ""
                Write-Host "To fix this:" -ForegroundColor Cyan
                Write-Host "  1. Open Docker Desktop from the Start Menu" -ForegroundColor Yellow
                Write-Host "  2. Wait for it to fully start (whale icon in system tray should be steady)" -ForegroundColor Yellow
                Write-Host "  3. Try running this script again" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "If Docker Desktop is not installed:" -ForegroundColor Gray
                Write-Host "  - Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Gray
                Write-Host ""
                Read-Host "Press Enter to exit"
                exit 1
            }
        } else {
            # Timeout occurred - Docker is not running
            Remove-Job -Job $job -Force
            Write-Host ""
            Write-Host "[ERROR] Docker is installed but NOT running!" -ForegroundColor Red
            Write-Host ""
            Write-Host "Docker Desktop is not currently running." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "To fix this:" -ForegroundColor Cyan
            Write-Host "  1. Open Docker Desktop from the Start Menu" -ForegroundColor Yellow
            Write-Host "  2. Wait for it to fully start (whale icon in system tray should be steady)" -ForegroundColor Yellow
            Write-Host "  3. Try running this script again" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "If Docker Desktop is not installed:" -ForegroundColor Gray
            Write-Host "  - Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Gray
            Write-Host ""
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Host ""
        Write-Host "[ERROR] Docker not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Docker is not installed or not in PATH" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To install Docker Desktop:" -ForegroundColor Cyan
        Write-Host "  - Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] Docker not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Docker is not installed or not in PATH" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Docker Desktop:" -ForegroundColor Cyan
    Write-Host "  - Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting Ollama Manager..." -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$managerScript = Join-Path $scriptDir "ollama_manager.py"

if (Test-Path $managerScript) {
    & $pythonCmd $managerScript
} else {
    Write-Host "Error: ollama_manager.py not found in $scriptDir" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
