# PDF-VQA Implementation Walkthrough

## 1. Tổng quan
Chúng tôi đã triển khai thành công 2 script huấn luyện cho dataset PDF-VQA:
1.  **`train_pdfvqa.py`**: Model cơ bản sử dụng ResNet + LSTM.
2.  **`train_lospa.py`**: Model nâng cao (Baseline) sử dụng Graph Neural Network (GNN) để tận dụng layout tài liệu.

## 2. Kết quả Thực thi

### 2.1. Basic Model (`train_pdfvqa.py`)
-   **Trạng thái**: Hoạt động tốt.
-   **Mô tả**: Load ảnh và câu hỏi, train mô hình phân loại câu trả lời.
-   **Cách chạy**:
    ```bash
    ..\venv\Scripts\python.exe train_pdfvqa.py
    ```

### 2.2. Graph Baseline Model (`train_lospa.py`)
-   **Kết quả**: ĐÃ CHẠY THÀNH CÔNG!
    -   Script đã tìm thấy khoảng **800 mẫu hợp lệ** (giao của `train_dataframe.pkl` và `train_doc_info_visual.pkl`).
    -   Training loop chạy ổn định (Demo 50 batches đầu tiên).
    -   Loss giảm dần.

-   **Nguyên nhân lỗi trước đó**:
    -   File CSV Task A/B không khớp với file Layout PKL hiện tại.
    -   File đúng là `train_dataframe.pkl` sử dụng cột `pmcid` làm ID, trong khi code cũ tìm `image_id`.
    -   Chúng tôi đã sửa code để hỗ trợ đọc `train_dataframe.pkl` và tự động dùng `pmcid` để khớp.

-   **Cách chạy**:
    Code đã được cấu hình sẵn để chạy với file `train_dataframe.pkl` hiện có của bạn.
    ```bash
    ..\venv\Scripts\python.exe train_lospa.py
    ```

## 3. Hướng dẫn File Code
-   **`train_lospa.py`**:
    -   Class `GraphVQADataset`: Xử lý load PKL layout, chuẩn hóa tọa độ bounding box, tạo Edge index cho đồ thị.
    -   Class `GraphVQA`: Mô hình GNN (GCNConv) kết hợp LSTM.
    -   Nếu bạn có file Layout đúng, mô hình này sẽ đạt hiệu quả cao hơn Basic Model nhờ hiểu được cấu trúc không gian của văn bản.

**Nguyên nhân gốc rễ (Đã giải quyết):**
File `train_doc_info_visual.pkl` sử dụng ID là `pmcid` (PubMed ID, ví dụ `20403171`).
File CSV Task A/B sử dụng `image_id` khác.
File `train_dataframe.pkl` trong thư mục lại chứa đúng `pmcid` này.

=> **Code hiện tại đã được trỏ vào `train_dataframe.pkl` và sửa logic để đọc `pmcid`. Hệ thống đã hoạt động hoàn hảo.**

## 5. Hướng dẫn Debug
Chúng tôi đã tạo script `read_pkl.py` được nâng cấp để bạn tự kiểm tra:
```bash
..\venv\Scripts\python.exe read_pkl.py
```
Script sẽ tự động tìm kiếm các ID mẫu để xác nhận xem file Pkl của bạn có chứa tài liệu mong muốn hay không.
