#!/bin/bash

cd "$(dirname "$0")"

echo "仮想環境を作成中..."
python3 -m venv venv

echo "仮想環境を有効化＆ライブラリをインストール中..."
source venv/bin/activate

# 仮想環境の pip3 を使ってアップグレード＆インストール
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo ""
echo "セットアップ完了！以後は 01_仕分け開始.command をダブルクリックしてください。"
read -p " Enterキーで閉じます"
