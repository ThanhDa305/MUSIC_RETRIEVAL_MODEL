import torch
import torchaudio
import numpy as np
import os
import json
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm
from pathlib import Path

# --- Cấu hình ---
MODEL_PATH = "./models"
MUSIC_LIBRARY_PATH = "./music_library"
INDEX_PATH = "./index"
TARGET_SAMPLING_RATE = 16000

# Kiểm tra và sử dụng GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tạo thư mục chứa index nếu chưa có
Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)

# --- 1. Tải Model và Feature Extractor đã fine-tuned ---
print("Đang tải model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()  # Chuyển sang chế độ inference
print("Tải model thành công!")


# --- 2. Hàm trích xuất vector từ file âm thanh ---
def extract_embedding(audio_path):
    try:
        # Tải và resample audio về 16kHz
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != TARGET_SAMPLING_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLING_RATE)
            waveform = resampler(waveform)

        # Chỉ lấy kênh đầu tiên nếu là stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Chuẩn bị input cho model
        inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt",
                                   padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Lấy embedding (không tính gradient để tiết kiệm VRAM)
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy vector đại diện bằng cách lấy trung bình các hidden state cuối cùng
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Lỗi khi xử lý file {audio_path}: {e}")
        return None


# --- 3. Quét toàn bộ thư viện nhạc và tạo index ---
print("Bắt đầu quá trình vector hóa kho nhạc...")
all_embeddings = []
all_file_paths = []

# Lấy danh sách các file audio được hỗ trợ
audio_files = list(Path(MUSIC_LIBRARY_PATH).rglob("*.mp3")) + \
              list(Path(MUSIC_LIBRARY_PATH).rglob("*.wav")) + \
              list(Path(MUSIC_LIBRARY_PATH).rglob("*.flac"))

for file_path in tqdm(audio_files, desc="Đang xử lý"):
    embedding = extract_embedding(str(file_path))
    if embedding is not None:
        all_embeddings.append(embedding)
        all_file_paths.append(str(file_path))

# Chuyển thành numpy array để tính toán hiệu quả
embeddings_matrix = np.array(all_embeddings)

# --- 4. Lưu index và danh sách file ---
np.save(os.path.join(INDEX_PATH, "embeddings.npy"), embeddings_matrix)

with open(os.path.join(INDEX_PATH, "file_paths.json"), "w", encoding='utf-8') as f:
    json.dump(all_file_paths, f, ensure_ascii=False, indent=4)

print(f"\nHoàn tất! Đã vector hóa {len(all_file_paths)} file nhạc.")
print(f"Index đã được lưu tại thư mục: {INDEX_PATH}")