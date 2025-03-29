import face_recognition
import os
import shutil

# === 設定 ===
BASE_DIR = 'face_sorter'
REFERENCE_DIR = os.path.join(BASE_DIR, 'reference_faces')
INPUT_DIR = os.path.join(BASE_DIR, 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_images')
UNKNOWN_DIR = os.path.join(OUTPUT_DIR, 'unknown')
THRESHOLD = 0.5

# === 初期準備 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

print("顔画像仕分けツールを開始します...\n")

# === 正解画像（reference_faces）を読み込み ===
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
        try:
            image = face_recognition.load_image_file(os.path.join(person_path, file))
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
        except Exception as e:
            print(f"[WARN] {file} 読み込み失敗 → {e}")

    if encodings:
        reference_faces[person_name] = encodings
        print(f"  - {person_name}: {len(encodings)}枚 登録完了")
    else:
        print(f"  - ⚠️ {person_name}: 顔登録できませんでした")

# === 画像仕分け処理 ===
print("\n画像の仕分けを開始します...\n")
failed_files = []

for file in os.listdir(INPUT_DIR):
    if file.startswith("."):
        continue

    input_path = os.path.join(INPUT_DIR, file)

    try:
        image = face_recognition.load_image_file(input_path)
        encodings = face_recognition.face_encodings(image)
    except Exception as e:
        failed_files.append(file)
        continue

    if not encodings:
        output_path = os.path.join(UNKNOWN_DIR, file)
        shutil.copy(input_path, output_path)
        print(f"  - {file}: 顔が検出できず → unknown")
        continue

    # 最小距離でマッチング（軽量化）
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

    # すでにあるならスキップ（重複防止）
    if os.path.exists(output_path):
        print(f"  - {file}: {best_match}（スキップ：すでに存在）")
        continue

    try:
        shutil.copy(input_path, output_path)
        print(f"  - {file}: {best_match}")
    except Exception as e:
        print(f"[ERROR] コピー失敗: {file} → {e}")
        failed_files.append(file)

# === ログ出力（失敗ファイル） ===
if failed_files:
    log_path = os.path.join(BASE_DIR, "copy_failed.log")
    with open(log_path, 'w') as f:
        for file in failed_files:
            f.write(file + '\n')
    print(f"\n⚠️ コピー失敗ファイルを {log_path} に保存しました（{len(failed_files)}件）")

print("\nすべての処理が完了しました！")
input("\nEnterキーを押すとウィンドウを閉じます")
