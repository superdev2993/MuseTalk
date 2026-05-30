#!/bin/bash
# Open MuseTalk web ports in UFW (requires sudo password).
set -e

PORTS="${PORTS:-7860 8080 8000}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo:"
  echo "  sudo $0"
  exit 1
fi

if command -v ufw >/dev/null 2>&1; then
  echo "=== UFW: allow MuseTalk ports ==="
  for port in $PORTS; do
    ufw allow "${port}/tcp" comment "MuseTalk web"
    echo "  allowed ${port}/tcp"
  done
  ufw status numbered | head -40
  echo ""
  echo "UFW rules updated. If status was inactive, enable with: sudo ufw enable"
else
  echo "ufw not found, trying iptables..."
  for port in $PORTS; do
    iptables -I INPUT -p tcp --dport "$port" -j ACCEPT
    echo "  iptables ACCEPT ${port}/tcp"
  done
fi

echo ""
echo "Done. Test from your browser:"
IP=$(hostname -I | awk '{print $1}')
echo "  http://${IP}:7860/"
