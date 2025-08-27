#!/bin/bash
# 🤖 Quick Start Script for Hive-Mind System
# One-command setup and launch

set -e

echo "🚀 Hive-Mind Lottery Prediction System - Quick Start"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $python_version"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating directories..."
mkdir -p data logs models/pytorch models/sklearn config

# Initialize database
print_status "Initializing database..."
if [ "$1" = "--reset-db" ]; then
    python scripts/init_database.py reset
else
    python scripts/init_database.py init
fi

# Verify installation
print_status "Verifying installation..."
python scripts/init_database.py verify

# Check if res.json exists
if [ ! -f "data/res.json" ]; then
    print_warning "Historical data file 'data/res.json' not found."
    print_warning "Please place your lottery data file in the data/ directory."
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Available commands:"
echo "  ${BLUE}Development server:${NC}    python scripts/startup.py server --env development"
echo "  ${BLUE}Production server:${NC}     python scripts/startup.py server --env production"
echo "  ${BLUE}Standalone mode:${NC}       python scripts/startup.py standalone"
echo "  ${BLUE}Demo predictions:${NC}      python scripts/startup.py demo"
echo "  ${BLUE}Docker compose:${NC}        docker-compose up -d"
echo ""
echo "API Documentation will be available at: http://localhost:8000/docs"
echo "System health check: http://localhost:8000/api/monitoring/health"
echo ""

# Ask user what to do next
read -p "Start development server now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting development server..."
    python scripts/startup.py server --env development
fi