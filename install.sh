#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/trade-bot"

echo "=== Installing EURUSD Rapid Scalper ==="

sudo apt update && sudo apt install -y python3 python3-pip python3-venv

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install yfinance pandas numpy

# Copy your files here (main.py, paper_trade.py, config.py, analysis/, execution/, risk/, signals/)

echo "Creating systemd services..."

# Live monitor service
sudo tee /etc/systemd/system/trade-live.service > /dev/null <<'SVC'
[Unit]
Description=Trade Bot - Live Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=CHANGE_ME
WorkingDirectory=CHANGE_ME
ExecStart=CHANGE_ME/venv/bin/python3 main.py
ExecStopPost=/bin/sleep 2
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SVC

# Paper trader service
sudo tee /etc/systemd/system/trade-paper.service > /dev/null <<'SVC'
[Unit]
Description=Trade Bot - Paper Trader (SimpleFX)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=CHANGE_ME
WorkingDirectory=CHANGE_ME
ExecStart=CHANGE_ME/venv/bin/python3 paper_trade.py
ExecStopPost=/bin/sleep 2
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SVC

echo ""
echo "=== Edit the service files to set your username/paths ==="
echo "  sudo nano /etc/systemd/system/trade-live.service"
echo "  sudo nano /etc/systemd/system/trade-paper.service"
echo ""
echo "Then start:"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable trade-live trade-paper"
echo "  sudo systemctl start trade-live trade-paper"
echo ""
echo "Monitor:"
echo "  sudo journalctl -u trade-live -f --no-hostname"
echo "  sudo journalctl -u trade-paper -f --no-hostname"
