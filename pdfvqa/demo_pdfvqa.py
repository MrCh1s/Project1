import pickle
import os
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ==========================================
# CẤU TRÚC CODE MẪU (Bổ sung Class Dataset)
# ==========================================
# Check PyTorch availability
try:
    from torch.utils.data import Dataset, DataLoader
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] Chưa cài đặt PyTorch via pip install torch. Code Dataset sẽ không chạy.")

def load_data(file_path):
    """Hàm đọc file data (pickle hoặc csv)"""
    if not os.path.exists(file_path):
        print(f"[LỖI] Không tìm thấy file: {file_path}")
        return None
    
    print(f"-> Đang tải file: {file_path}...")
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        print("[LỖI] Định dạng file không hỗ trợ (chỉ .pkl hoặc .csv)")
        return None

    print(f"-> Tải thành công! Kích thước dữ liệu: {len(data)}")
    return data

class PDFVQADataset(Dataset):
    def __init__(self, qa_file, layout_file, image_dir, transform=None):
        """
        Args:
            qa_file (str): Đường dẫn đến file QA (.pkl hoặc .csv).
            layout_file (str): Đường dẫn đến file .pkl chứa layout structure.
            image_dir (str): Thư mục chứa ảnh gốc.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load QA Data
        self.qa_data = load_data(qa_file)
        
        # Load Layout Data
        self.layout_data = load_data(layout_file)
            
        # Xử lý chuẩn hóa danh sách QA
        if isinstance(self.qa_data, pd.DataFrame):
            # Nếu là CSV (convert sang list dict)
            # CSV headers: ,Unnamed: 0,answer,file_name,type,question
            # file_name ví dụ: 29880484.pdf_5.png chnh là image_id (hoặc chứa image_id)
            self.qa_list = self.qa_data.to_dict('records')
        elif isinstance(self.qa_data, dict):
            self.qa_list = list(self.qa_data.values())
        else:
            self.qa_list = self.qa_data

    def __len__(self):
        return len(self.qa_list)

    def __getitem__(self, idx):
        qa_item = self.qa_list[idx]
        
        # Xử lý lấy image_id
        # Các file CSV/PKL có tên cột khác nhau: 'image_id', 'file_name', 'file', ...
        image_id = str(qa_item.get('image_id', ''))
        
        if not image_id or image_id == 'nan':
            # Thử các key phổ biến khác
            for k in ['file_name', 'file', 'image_name']:
                if k in qa_item and str(qa_item[k]) != 'nan':
                     image_id = str(qa_item[k])
                     break
        
        # Lấy Layout (Thử trực tiếp image_id hoặc bỏ đuôi .png)
        # Trong CSV Task B, file có dạng "29880484.pdf_5.png"
        layout_info = self.layout_data.get(image_id, {})
        if not layout_info:
             # Thử bỏ đuôi ảnh (vd: .png) phòng trường hợp key trong pkl không có đuôi
             core_id = image_id.rsplit('.', 1)[0]
             layout_info = self.layout_data.get(core_id, {})

        # Load ảnh
        img_path = os.path.join(self.image_dir, image_id)
        # Fallback nếu tên trong QA không có đuôi nhưng file ảnh có đuôi
        if not os.path.exists(img_path) and not image_id.endswith(('.jpg', '.png')):
             if os.path.exists(img_path + ".jpg"): img_path += ".jpg"
             elif os.path.exists(img_path + ".png"): img_path += ".png"
             
        image = None
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        
        return {
            "question": qa_item.get('question'),
            "answer": qa_item.get('answer'),
            "image": image,
            "layout": layout_info, 
            "image_id": image_id
        }

def main():
    print("--- HƯỚNG DẪN SỬ DỤNG ---")
    print("Code hỗ trợ tự động tìm file .pkl (gốc) HOẶC .csv")
    print("-" * 30)

    # === CẤU HÌNH TASK ===
    # Bạn đổi tên task ở đây: "TaskA", "TaskB", "TaskC"
    # Hoặc tên file cụ thể nếu muốn.
    TARGET_TASK = "TaskB" 
    # =====================

    possible_roots = ["..", "./data", "."]
    base_dir = None
    for r in possible_roots:
        if os.path.exists(os.path.join(r, "train_images")):
            base_dir = r
            break
    
    if not base_dir:
        base_dir = ".." 
        print("[INFO] Không tìm thấy train_images. Giả định path là '..'")

    print(f"[INFO] Thư mục dữ liệu: {os.path.abspath(base_dir)}")
    
    # 1. Tìm file QA dựa trên CONFIG
    # Code sẽ ưu tiên tìm file CSV/PKL khớp với tên Task
    qa_filename_candidates = [
        f"pdfvqa_{TARGET_TASK.lower()}_train_0509_clean.csv", # Task A pattern
        f"pdfvqa_{TARGET_TASK.lower()}_train_0503_clean.csv", # Task B pattern
        f"train_dataframe.pkl",                                # Task C pattern (đặc biệt)
        f"train_{TARGET_TASK}_qa.pkl",                         # Generic pattern
    ]
    
    qa_path = None
    # Logic tìm file: 
    # Nếu là Task C thì ưu tiên tìm train_dataframe.pkl
    if TARGET_TASK == "TaskC":
        c_path = os.path.join(base_dir, "train_dataframe.pkl")
        if os.path.exists(c_path):
            qa_path = c_path
            
    if not qa_path:
        for fname in qa_filename_candidates:
            p = os.path.join(base_dir, fname)
            if os.path.exists(p):
                # Chỉ nhận nếu tên file khớp với Task đang chọn (để tránh load nhầm Task A khi đang chọn Task B)
                # Tuy nhiên với file list trên user thì tên file đã khá rõ ràng.
                # Riêng csv Task A và B có tên gần giống nhau (chỉ khác date 0509 vs 0503 và taska/taskb)
                if TARGET_TASK.lower() in fname.lower() or (TARGET_TASK == "TaskC" and "dataframe" in fname):
                    qa_path = p
                    break
    
    # Fallback cũ
    if not qa_path:
         qa_path = os.path.join(base_dir, "pdfvqa_taska_train_0509_clean.csv")

    # 2. Tìm file Layout (Dùng chung cho mọi task)
    layout_path = os.path.join(base_dir, "train_doc_info_visual.pkl")
    if not os.path.exists(layout_path):
        layout_path = os.path.join(base_dir, "train_layout.pkl")
    
    img_dir = os.path.join(base_dir, "train_images")

    print(f"Path QA: {os.path.basename(qa_path)} | Exists: {os.path.exists(qa_path)}")
    print(f"Path Layout: {os.path.basename(layout_path)} | Exists: {os.path.exists(layout_path)}")

    if not os.path.exists(qa_path) or not os.path.exists(layout_path):
        print(f"[CẢNH BÁO] Thiếu file dữ liệu!")
        return

    if HAS_TORCH:
        try:
            print("Đang khởi tạo Dataset...")
            dataset = PDFVQADataset(qa_path, layout_path, img_dir)
            print(f"Tổng số mẫu: {len(dataset)}")
            
            if len(dataset) > 0:
                import random
                # Chọn ngẫu nhiên 3 mẫu để hiển thị
                indices = random.sample(range(len(dataset)), min(3, len(dataset)))
                
                print(f"\n[INFO] Đang hiển thị {len(indices)} mẫu ngẫu nhiên...")
                for i, idx in enumerate(indices):
                    print(f"\n=== Mẫu thử #{i+1} (Index: {idx}) ===")
                    sample = dataset[idx]
                    print("Image ID:", sample['image_id'])
                    print("Câu hỏi:", sample['question'])
                    print("Đáp án:", sample['answer'])
                    
                    # Gọi hàm visualize (Lưu ý: sẽ pop-up cửa sổ ảnh, bạn cần đóng để xem ảnh tiếp theo)
                    visualize_sample(sample)
        except Exception as e:
            print(f"Lỗi khi chạy dataset: {e}")
            if "pandas" in str(e):
                 print("-> Hãy cài đặt pandas: pip install pandas")
    else:
        print("Cài đặt PyTorch để chạy thử: pip install torch")

def visualize_sample(sample):
    """
    Hàm hiển thị ảnh và bounding boxes của sample.
    """
    image = sample['image']
    layout = sample['layout']
    
    if image is None:
        print("[VIS] Không tìm thấy ảnh để hiển thị.")
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Layout thường chứa 'bboxes' hoặc các key tương tự. 
    # Cấu trúc layout_info trong PDF-VQA có thể thay đổi tùy file pkl, cần inspect kỹ.
    # Code này giả định structure phổ biến: layout[id] = {'bboxes': [[x1,y1,x2,y2], ...], 'texts': [...]}
    if layout and isinstance(layout, dict):
        bboxes = layout.get('bboxes', [])
        texts = layout.get('texts', [])
        
        for i, box in enumerate(bboxes):
            # Giả sử box là [x1, y1, x2, y2]
            if len(box) == 4:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Có thể vẽ thêm text index nếu muốn
                # plt.text(x1, y1, str(i), color='blue', fontsize=8)
    
    plt.title(f"Question: {sample['question']}\nAnswer: {sample['answer']}")
    plt.axis('off')
    print("[INFO] Đang hiển thị ảnh minh họa...")
    plt.show()

if __name__ == "__main__":
    # Check dependencies check
    try:
        import matplotlib
        import PIL
    except ImportError as e:
        print(f"[LỖI] Thiếu thư viện: {e.name}")
        print("Vui lòng cài đặt: pip install matplotlib pillow torch")
    
    main()
    
    # Nếu main chạy xong và có dataset, mình có thể demo visualize ở đây (cần sửa main để return dataset hoặc gọi visualize trong main)
    # Tuy nhiên để đơn giản, ta sẽ sửa main để gọi visualize_sample luôn.
