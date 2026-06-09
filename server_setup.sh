#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/trade-bot"
echo "=== Installing EURUSD Rapid Scalper on Ubuntu ==="

# 1. System deps
sudo apt update && sudo apt install -y python3 python3-pip python3-venv

# 2. Create project directory
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# 3. Python venv + deps
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install yfinance pandas numpy

# 4. Copy project files
echo "Copy your project files to $PROJECT_DIR then run:"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo "  python3 main.py --reset 100.0"
echo ""
echo "5. Start monitors:"
echo "  nohup bash -c 'while true; do python3 main.py >> logs/system_output.log 2>&1; sleep 2; done' > /dev/null 2>&1 &"
echo "  nohup bash -c 'while true; do python3 paper_trade.py >> logs/paper_trade_output.log 2>&1; sleep 2; done' > /dev/null 2>&1 &"
