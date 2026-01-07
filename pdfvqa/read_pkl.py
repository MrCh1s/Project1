import pickle
import sys
import os

def read_pickle_file(file_path):
    print(f"-> Đang tải file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print("Đã load xong!")
        print(f"Loại dữ liệu: {type(data)}")
        
        # Nếu là DataFrame (pandas)
        if hasattr(data, 'head'):
            print("Detected Pandas DataFrame")
            print(f"Bảng có {len(data)} dòng.")
            print(f"Columns: {data.columns.tolist()}")
            print("-" * 20)
            print("First 5 rows:")
            print(data.head())
            print("-" * 20)
            
            # Check for 'image_id' column
            if 'image_id' in data.columns:
                print("Samples of image_id:")
                print(data['image_id'].head())
                
        # Nếu là Dictionary (Từ điển)
        elif isinstance(data, dict):
            keys = list(data.keys())
            print(f"Số lượng keys: {len(keys)}")
            print(f"5 keys đầu tiên: {keys[:5]}")
            
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    # Thay đổi tên file tại đây:
    file_name = "train_dataframe.pkl"
    
    # Logic tìm file ở thư mục cha nếu không thấy
    if not os.path.exists(file_name) and os.path.exists(os.path.join("..", file_name)):
        file_name = os.path.join("..", file_name)
        
    read_pickle_file(file_name)