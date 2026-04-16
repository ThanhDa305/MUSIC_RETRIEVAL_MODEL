import torch
import torchaudio
import numpy as np
import json
import os
from transformers import AutoFeatureExtractor, AutoModel
from scipy.spatial.distance import cdist
# Thêm vào đầu file search.py
import wave
import pyaudio


# Hàm thu âm một đoạn ngắn
def record_audio(duration=7, save_path="temp_query.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"* Bắt đầu thu âm {duration} giây...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* Kết thúc thu âm.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(save_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return save_path


# --- Cấu hình ---
MODEL_PATH = "./models"
INDEX_PATH = "./index"
TARGET_SAMPLING_RATE = 16000
TOP_K = 5  # Số lượng kết quả trả về

# --- 1. Tải Model và các tài nguyên đã xử lý ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tải model và feature extractor
print("Đang tải model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("Tải model thành công.")

# Tải index đã được tạo
print("Đang tải index...")
embeddings_matrix = np.load(os.path.join(INDEX_PATH, "embeddings.npy"))
with open(os.path.join(INDEX_PATH, "file_paths.json"), "r", encoding='utf-8') as f:
    file_paths = json.load(f)
print(f"Tải index thành công. {len(file_paths)} bài hát trong cơ sở dữ liệu.")


# --- 2. Hàm trích xuất vector (giống file create_index.py) ---
def extract_embedding(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != TARGET_SAMPLING_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLING_RATE)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt",
                                   padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Lỗi khi xử lý file {audio_path}: {e}")
        return None


# --- 3. Hàm tìm kiếm ---
def search(query_path, top_k=TOP_K):
    print(f"\nĐang tìm kiếm cho file: {query_path}")

    # Trích xuất vector của file query
    query_embedding = extract_embedding(query_path)
    if query_embedding is None:
        print("Không thể xử lý file query.")
        return

    # Tính toán Cosine Similarity
    distances = cdist(query_embedding, embeddings_matrix, metric='cosine')[0]

    # Lấy top_k index có khoảng cách nhỏ nhất
    top_k_indices = np.argsort(distances)[:top_k]

    # In kết quả
    print(f"--- Top {top_k} kết quả tương đồng nhất ---")
    for i, idx in enumerate(top_k_indices):
        similarity = 1 - distances[idx]
        file_name = os.path.basename(file_paths[idx])
        print(f"{i + 1}. {file_name} (Độ tương đồng: {similarity:.4f})")


# --- 4. Chạy thử ---
if __name__ == "__main__":
    # Lựa chọn 2: Thu âm và tìm kiếm (ĐANG KÍCH HOẠT)
    temp_file = record_audio(duration=7)
    search(temp_file)