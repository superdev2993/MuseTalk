#!/bin/bash
# Disable UFW entirely (requires sudo). Use only for local testing.
set -e

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo:"
  echo "  sudo $0"
  exit 1
fi

if command -v ufw >/dev/null 2>&1; then
  ufw disable
  ufw status
  echo "UFW disabled."
else
  echo "ufw not installed."
fi
