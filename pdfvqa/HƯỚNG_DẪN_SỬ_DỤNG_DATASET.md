# Hướng dẫn Cài đặt và Sử dụng PDF-VQA Dataset

Sau khi bạn đã tải các thư mục ảnh (`train_images`, `val_images`, `test_images`), bạn cần tải thêm các file metadata (.pkl) để có thể sử dụng.

## 1. Tải Dữ liệu Annotations (.pkl)
Bạn hãy tải các file dưới đây về:

### A. Layout Structure (Bắt buộc cho mọi Task)
Chứa thông tin Bounding Box, Text, và Relations.
- **Training**: [Tải train_layout.pkl](https://drive.google.com/file/d/1SyEptlqqX-frq_1hSQTxUGGptk6OI9aQ/view?usp=drive_link)
- **Validation**: [Tải val_layout.pkl](https://drive.google.com/file/d/1Z9umISob9ar_5n5T-Cbhr4nbHuQCvVGm/view?usp=drive_link)
- **Testing**: [Tải test_layout.pkl](https://drive.google.com/file/d/1knSVmocw4-_FF98bFMdVSvhnUn3mPUvm/view?usp=drive_link)

### B. Question-Answer Pairs (Chọn Task)
Tùy vào bạn muốn làm Task nào, hãy tải file tương ứng:

**Task A (Element Recognition)**
- [Train Task A](https://drive.google.com/file/d/19UuBpZDXmyiU7oFkJaOPTYGtFMkT8b6-/view?usp=drive_link) | [Val Task A](https://drive.google.com/file/d/1gGIzsSZHVokehACx7h-SOk5K1uEEXNpq/view?usp=drive_link) | [Test Task A](https://drive.google.com/file/d/1HIYxpGCcXdQo42b79Eqfmji9U9-YmUjp/view?usp=drive_link)

**Task B (Structural Understanding)**
- [Train Task B](https://drive.google.com/file/d/1FJ8haTaHKEe0LX1HVVY1fIk8ogLv0GVw/view?usp=drive_link) | [Val Task B](https://drive.google.com/file/d/18N7YACPIvPQ1Mr1Lvmh4ohKAYgZWDfYp/view?usp=drive_link) | [Test Task B](https://drive.google.com/file/d/1FrAB0tKcVg3r67yi-Q2pDw928ACxeRzH/view?usp=drive_link)

**Task C (Document Understanding)**
- [Train Task C](https://drive.google.com/file/d/1-8ECkZV5q5i7CBY7U1llmCTSiVJPUNsS/view?usp=drive_link) | [Val Task C](https://drive.google.com/file/d/1-WdGXwo_8U7cU8mOh5ZjLm_CDqDDWbxo/view?usp=drive_link) | [Test Task C](https://drive.google.com/file/d/1-F242FFvubAIpjXPItFc_eGUs3dzb6QO/view?usp=drive_link)

## 2. Tổ chức Thư mục (KHUYẾN NGHỊ)
**KHÔNG** để file `.pkl` vào bên trong các thư mục `_images`. Hãy để chúng **ngang hàng** với nhau.

Cấu trúc chuẩn như sau:

```
Project1/
├── pdfvqa/                 # Thư mục code của bạn
├── train_images/           # CHỈ CHỨA ẢNH (.jpg/.png)
├── val_images/             # CHỈ CHỨA ẢNH
├── test_images/            # CHỈ CHỨA ẢNH
├── train_layout.pkl        # File Layout (để ở ngoài)
├── val_layout.pkl
├── train_taskA_qa.pkl      # File QA (để ở ngoài)
└── val_taskA_qa.pkl
```

Việc này giúp quản lý dễ dàng hơn và tránh lỗi khi code quét file ảnh trong thư mục.

## 3. Chạy Demo
Chạy lệnh sau để kiểm tra dữ liệu:
```bash
python demo_pdfvqa.py
```
Code trong `demo_pdfvqa.py` đã được cấu hình để đọc cấu trúc thư mục trên.
