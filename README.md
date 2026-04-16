🎵 Music Retrieval System (Vector Search)
Dự án tìm kiếm âm nhạc dựa trên đặc trưng (embeddings) sử dụng Deep Learning. Hệ thống cho phép tìm kiếm các bài hát tương đồng trong cơ sở dữ liệu dựa trên một đoạn nhạc truy vấn.

🚀 Tính năng chính
Trích xuất đặc trưng: Sử dụng model Pre-trained (như PANNs hoặc Wav2Vec) để chuyển đổi âm thanh sang vector.

Tìm kiếm Vector: Sử dụng tìm kiếm tương đồng (Cosine Similarity/Euclidean Distance) để tìm kết quả nhanh chóng.

Quản lý Index: Lưu trữ và tải chỉ mục vector hiệu quả.

🛠 Công nghệ sử dụng
Ngôn ngữ: Python 3.13+

Deep Learning: PyTorch, Torchaudio

Xử lý dữ liệu: NumPy, Librosa

Quản lý mã nguồn: Git LFS

📦 Cài đặt
Clone project:

Bash

git clone https://github.com/ThanhDa305/MUSIC_RETRIEVAL_MODEL.git
cd MUSIC_RETRIEVAL
Cài đặt môi trường ảo và thư viện:

Bash

python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
📂 Cấu trúc thư mục
Plaintext

MUSIC_RETRIEVAL/
├── index/              # Chứa file embedding (.npy) và mapping (.json)
├── models/             # Thư mục chứa file trọng số model (.pth)
├── music_library/      # Thư mục chứa dataset âm thanh (.wav)
├── create_index.py     # Script tạo index từ bộ thư viện nhạc
├── search.py           # Script thực hiện truy vấn tìm kiếm
└── temp_query.wav      # File nhạc tạm dùng để test
⚠️ Lưu ý quan trọng (Dataset & Weights)
Do giới hạn dung lượng của GitHub, các file trọng số và bộ dữ liệu nhạc đầy đủ đã được lược bỏ. Để chạy được dự án, bạn vui lòng thực hiện:

Tải file trọng số model tại: [Link]

Tải bộ dữ liệu nhạc (8GB) tại: [Link]

Giải nén và đặt vào đúng thư mục models/ và music_library/.

🖥 Hướng dẫn sử dụng
Tạo Index: (Chạy lần đầu để trích xuất vector cho bộ nhạc)

Bash

python create_index.py
Tìm kiếm nhạc:

Bash

python search.py
👤 Tác giả
Nguyễn Thanh Đa - Sinh viên ngành Hệ thống thông tin, Đại học Cần Thơ.
