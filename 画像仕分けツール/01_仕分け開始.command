#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
echo "Python 実行パス:${which python3}"
python3 01_仕分け開始.py
