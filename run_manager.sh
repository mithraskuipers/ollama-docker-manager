#!/bin/bash
# Ollama Docker Manager - Linux/WSL Launcher
# Cross-platform launcher for Ollama Docker Manager

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo ""
echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}           OLLAMA DOCKER MANAGER - Cross Platform${NC}"
echo -e "${CYAN}=================================================================${NC}"
echo ""

# Check if Python is installed
echo -e "${CYAN}Checking if Python is installed...${NC}"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        VERSION=$($cmd --version 2>&1)
        echo -e "${GREEN}✓ Found Python: $VERSION${NC}"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}✗ Python not found!${NC}"
    echo ""
    echo -e "${YELLOW}Please install Python 3.7 or later:${NC}"
    echo -e "${CYAN}  Ubuntu/Debian: sudo apt install python3${NC}"
    echo -e "${CYAN}  Fedora/RHEL:   sudo dnf install python3${NC}"
    echo -e "${CYAN}  Arch:          sudo pacman -S python${NC}"
    echo ""
    exit 1
fi

# Check for requests library
echo ""
echo -e "${CYAN}Checking Python dependencies...${NC}"
if $PYTHON_CMD -c "import requests" 2>/dev/null; then
    echo -e "${GREEN}✓ Python 'requests' library found${NC}"
else
    echo -e "${YELLOW}⚠ Python 'requests' library not found${NC}"
    echo ""
    echo -e "${CYAN}Installing 'requests' library...${NC}"
    if $PYTHON_CMD -m pip install requests --user 2>/dev/null; then
        echo -e "${GREEN}✓ Successfully installed 'requests'${NC}"
    else
        echo -e "${YELLOW}Note: The chat feature requires the 'requests' library${NC}"
        echo -e "${CYAN}Install manually with: $PYTHON_CMD -m pip install requests${NC}"
        echo -e "${CYAN}Or: sudo apt install python3-requests${NC}"
        echo ""
    fi
fi

# Check Docker
if command -v docker &> /dev/null; then
    # Docker command exists, now check if daemon is running (with timeout)
    if timeout 5 docker info &> /dev/null 2>&1; then
        DOCKER_VERSION=$(docker --version 2>&1)
        echo -e "${GREEN}✓ Found Docker: $DOCKER_VERSION${NC}"
        echo -e "${GREEN}✓ Docker daemon is running${NC}"
    else
        echo -e "${RED}✗ Docker is installed but NOT running!${NC}"
        echo ""
        echo -e "${YELLOW}Docker daemon is not currently running.${NC}"
        echo ""
        echo -e "${CYAN}To fix this:${NC}"
        
        # Check if we're in WSL
        if grep -qi microsoft /proc/version 2>/dev/null; then
            echo -e "${YELLOW}  WSL detected:${NC}"
            echo "    1. Open Docker Desktop in Windows (from Start Menu)"
            echo "    2. Wait for it to fully start (icon in system tray should be green)"
            echo "    3. Enable WSL2 integration for your distro:"
            echo "       Settings → Resources → WSL Integration → Ubuntu"
            echo "       (toggle it ON, then click 'Apply & Restart')"
        else
            echo -e "${YELLOW}  Linux:${NC}"
            echo "    Run: sudo systemctl start docker"
            echo "    Or:  sudo service docker start"
        fi
        echo ""
        echo -e "${RED}Please start Docker and try again.${NC}"
        echo ""
        exit 1
    fi
else
    echo -e "${RED}✗ Docker not found!${NC}"
    echo ""
    echo -e "${YELLOW}Docker is not installed or not in PATH${NC}"
    echo ""
    echo -e "${CYAN}To install Docker:${NC}"
    echo "  Ubuntu/Debian: sudo apt install docker.io"
    echo "  Or visit: https://docs.docker.com/engine/install/"
    echo ""
    exit 1
fi

echo ""
echo -e "${CYAN}Starting Ollama Manager...${NC}"
echo ""

# Get script directory and run Python manager
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MANAGER_SCRIPT="$SCRIPT_DIR/ollama_manager.py"

if [ -f "$MANAGER_SCRIPT" ]; then
    $PYTHON_CMD "$MANAGER_SCRIPT"
else
    echo -e "${RED}✗ Error: ollama_manager.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi