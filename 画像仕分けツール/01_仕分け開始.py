import face_recognition
import os
import shutil
from PIL import Image
import numpy as np

# === スクリプトの場所を起点にパス構成 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'face_sorter')
REFERENCE_DIR = os.path.join(BASE_DIR, 'reference_faces')
INPUT_DIR = os.path.join(BASE_DIR, 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_images')
UNKNOWN_DIR = os.path.join(OUTPUT_DIR, 'unknown')
THRESHOLD = 0.5
RESIZE_MAX = 1200  # 対象画像だけリサイズ

# === 初期準備 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

print("顔画像仕分けツールを開始します...\n")

# === 正解画像を読み込み（リサイズなし） ===
reference_faces = {}
print("顔データを読み込み中...")

for person_name in os.listdir(REFERENCE_DIR):
    person_path = os.path.join(REFERENCE_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    encodings = []
    for file in os.listdir(person_path):
        if file.startswith("."):
            continue
        file_path = os.path.join(person_path, file)
        try:
            pil_image = Image.open(file_path).convert("RGB")
            image_np = np.array(pil_image)  # 正解データはリサイズしない
            face_locations = face_recognition.face_locations(image_np)
            if not face_locations:
                print(f"[SKIP] 顔が検出できませんでした → {file_path}")
                continue
            face_encs = face_recognition.face_encodings(image_np, face_locations)
            if face_encs:
                encodings.append(face_encs[0])
        except Exception as e:
            print(f"[WARN] 読み込み失敗: {file_path} → {e}")

    if encodings:
        reference_faces[person_name] = encodings
        print(f"  - {person_name}: {len(encodings)}枚 登録完了")
    else:
        print(f"  - {person_name}: 顔登録できませんでした")

# === 画像仕分け処理（対象画像はリサイズあり） ===
print("\n画像の仕分けを開始します...\n")
failed_files = []

for file in os.listdir(INPUT_DIR):
    if file.startswith("."):
        continue

    input_path = os.path.join(INPUT_DIR, file)

    try:
        pil_image = Image.open(input_path).convert("RGB")
        pil_image.thumbnail((RESIZE_MAX, RESIZE_MAX))  # 仕分ける時にリサイズあり
        image_np = np.array(pil_image)
        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            output_path = os.path.join(UNKNOWN_DIR, file)
            shutil.copy(input_path, output_path)
            print(f"  - {file}: 顔が検出できず → unknown")
            continue
        encodings = face_recognition.face_encodings(image_np, face_locations)
    except Exception as e:
        failed_files.append(file)
        print(f"[ERROR] {file} の処理中にエラー: {e}")
        continue

    if not encodings:
        output_path = os.path.join(UNKNOWN_DIR, file)
        shutil.copy(input_path, output_path)
        print(f"  - {file}: 顔が検出できず → unknown")
        continue

    target_encoding = encodings[0]
    best_match = 'unknown'
    best_distance = float('inf')

    for name, ref_encs in reference_faces.items():
        distances = face_recognition.face_distance(ref_encs, target_encoding)
        min_distance = min(distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_match = name

    if best_distance < THRESHOLD:
        person_dir = os.path.join(OUTPUT_DIR, best_match)
    else:
        person_dir = UNKNOWN_DIR
        best_match = "unknown"

    os.makedirs(person_dir, exist_ok=True)
    output_path = os.path.join(person_dir, file)

    if os.path.exists(output_path):
        print(f"  - {file}: {best_match}（スキップ：すでに存在）")
        continue

    try:
        shutil.copy(input_path, output_path)
        print(f"  - {file}: {best_match}")
    except Exception as e:
        print(f"[ERROR] コピー失敗: {file} → {e}")
        failed_files.append(file)

# === ログ出力 ===
if failed_files:
    log_path = os.path.join(BASE_DIR, "copy_failed.log")
    with open(log_path, 'w') as f:
        for file in failed_files:
            f.write(file + '\n')
    print(f"\nコピー失敗ファイルを {log_path} に保存しました（{len(failed_files)}件）")

print("\nすべての処理が完了しました。")
input("\nEnterキーを押すとウィンドウを閉じます")
