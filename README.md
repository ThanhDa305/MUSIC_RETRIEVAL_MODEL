<p align="center">🎵 MUSIC RETRIEVAL SYSTEM 🎵</p><p align="center"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" /><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" /><img src="https://img.shields.io/badge/Vector%20Search-FF6F00?style=for-the-badge&logo=databricks&logoColor=white" /><img src="https://img.shields.io/badge/Deep%20Learning-00599C?style=for-the-badge&logo=google-cloud&logoColor=white" /></p><p align="center"><i>"Tìm kiếm giai điệu trong biển dữ liệu âm thanh"</i></p>🌟 OverviewDự án này là hệ thống Content-based Music Retrieval, cho phép người dùng tìm kiếm bài hát bằng cách đưa vào một đoạn nhạc mẫu. Hệ thống sẽ trích xuất "dấu vân tay âm thanh" (Embeddings) và so khớp trong không gian vector.🧠 Core LogicHệ thống sử dụng khoảng cách Cosine để tính toán độ tương đồng giữa vector truy vấn ($q$) và các vector trong cơ sở dữ liệu ($v$):$$\text{similarity} = \cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{v}}{\|\mathbf{q}\| \|\mathbf{v}\|}$$🛠 Tech StackAudio Processing: Librosa, TorchaudioModel: Pre-trained Convolutional Neural Networks (CNNs) / TransformersDatabase: NumPy indexing & JSON mappingEnvironment: PyCharm, Python 3.13📂 Project StructureBashMUSIC_RETRIEVAL/
├── 🗂 index/              # Lưu trữ "linh hồn" của hệ thống (Vector data)
├── 🧠 models/             # Nơi trú ngụ của các Weight khổng lồ
├── 🎵 music_library/      # Thư viện âm thanh (Dataset)
├── 📝 create_index.py     # Script trích xuất đặc trưng
├── 🔍 search.py           # Engine tìm kiếm cốt lõi
└── ⚡ requirements.txt     # "Bản đồ" thư viện
⚙️ Quick Start1. Triệu hồi môi trườngBashgit clone https://github.com/ThanhDa305/MUSIC_RETRIEVAL_MODEL.git
python -m venv .venv
# Activate nó lên và...
pip install -r requirements.txt
2. Dữ liệu nặng (Models & Music)[!IMPORTANT]Do chính sách "giảm cân" của dự án, các file nặng đã được cất giữ riêng.📥 Download Model & Dataset: [👉 Nhấn vào đây để tải (Google Drive)]Sau đó giải nén vào đúng thư mục ./models và ./music_library.🎬 Demo(Ông chèn một cái ảnh GIF quay cảnh terminal đang chạy hoặc kết quả tìm kiếm vào đây cho nó "pro")👨‍💻 AuthorNguyễn Thanh Đa🎓 Major: Information Systems @ Can Tho University🎯 Interest: Computer Vision, Deep Learning, AI in Medicine📫 Contact: [Email của ông] | [LinkedIn của ông]
