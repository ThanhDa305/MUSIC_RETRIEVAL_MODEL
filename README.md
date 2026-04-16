# <p align="center">🎵 MUSIC RETRIEVAL SYSTEM 🎵</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Vector%20Search-FF6F00?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep%20Learning-00599C?style=for-the-badge&logo=google-cloud&logoColor=white" />
</p>

<p align="center">
  <i>"Tìm kiếm giai điệu trong biển dữ liệu âm thanh"</i>
</p>

---

## 🌟 Overview
Dự án này là hệ thống **Content-based Music Retrieval**, cho phép người dùng tìm kiếm bài hát bằng cách đưa vào một đoạn nhạc mẫu. Hệ thống sẽ trích xuất "dấu vân tay âm thanh" (Embeddings) và so khớp trong không gian vector.

### 🧠 Core Logic
Hệ thống sử dụng khoảng cách Cosine để tính toán độ tương đồng giữa vector truy vấn $q$ và các vector trong cơ sở dữ liệu $v$:

$$\text{similarity} = \cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{v}}{\|\mathbf{q}\| \|\mathbf{v}\|}$$

---

## 🛠 Tech Stack
- **Audio Processing:** `Librosa`, `Torchaudio`
- **Model:** Pre-trained Convolutional Neural Networks (CNNs) / Transformers
- **Database:** `NumPy` indexing & `JSON` mapping
- **Environment:** PyCharm, Python 3.13

---

## 📂 Project Structure
```text
MUSIC_RETRIEVAL/
├──  index/              # Lưu trữ của hệ thống (Vector data)
├──  models/             # Nơi lưu trữ Weight khổng lồ
├──  music_library/      # Thư viện âm thanh (Dataset)
├──  create_index.py     # Script trích xuất đặc trưng
├──  search.py           # Engine tìm kiếm cốt lõi
└──  requirements.txt     # "Bản đồ" thư viện

git clone [https://github.com/ThanhDa305/MUSIC_RETRIEVAL_MODEL.git](https://github.com/ThanhDa305/MUSIC_RETRIEVAL_MODEL.git)
cd MUSIC_RETRIEVAL
python -m venv .venv
# Activate nó lên (Windows)
.\.venv\Scripts\activate
# Cài đặt thư viện
pip install -r requirements.txt

👨‍💻 Author
Nguyễn Thanh Đa

🎓 Major: Information Systems @ Can Tho University

🎯 Interest: Computer Vision, Deep Learning, AI in Medicine

📫 Contact: [nguyendapro2018@gmail.com] | [https://www.google.com/search?q=LinkedIn.com/in/dathanh305]
