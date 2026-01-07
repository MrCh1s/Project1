import fitz  # Đảm bảo không còn lỗi ở dòng này

def test_extraction(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Lấy cấu trúc phân cấp (Blocks) - Đây là tiền đề cho Relational Graph
        blocks = page.get_text("blocks")
        
        print(f"--- Đã đọc thành công: {pdf_path} ---")
        for b in blocks[:5]: # In thử 5 khối đầu tiên
            # b = (x0, y0, x1, y1, "nội dung", block_no, block_type)
            print(f"Block {b[5]} [{b[6]}]: {b[4][:50].strip()}...")
            
    except Exception as e:
        print(f"Lỗi: {e}")
