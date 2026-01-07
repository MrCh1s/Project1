import easyocr
import cv2
import json
import os

# 1. Khởi tạo EasyOCR cho Tiếng Việt và Tiếng Anh
print("Chương trình đang khởi tạo mô hình OCR (lần đầu có thể tốn thời gian)...")
reader = easyocr.Reader(['vi', 'en']) 

def process_image_to_graph(image_path, output_viz_path='vis_result.jpg'):
    # 2. Đọc hình ảnh để vẽ trực quan hóa
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh tại: {image_path}")
        return None

    # 3. Chạy OCR để lấy dữ liệu
    # result dạng: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "text", confidence), ...]
    print(f"Đang chạy OCR trên ảnh: {image_path}...")
    ocr_result = reader.readtext(image_path)

    # 4. Chuyển đổi kết quả OCR thành danh sách các Nodes (giống format PDF)
    nodes = []
    for i, (bbox, text, prob) in enumerate(ocr_result):
        # EasyOCR trả về bbox là 4 góc, ta chuyển về dạng chuẩn [x0, y0, x1, y1]
        x0 = int(min(point[0] for point in bbox))
        y0 = int(min(point[1] for point in bbox))
        x1 = int(max(point[0] for point in bbox))
        y1 = int(max(point[1] for point in bbox))
        
        nodes.append({
            "id": i,
            "text": text.strip(),
            "bbox": [x0, y0, x1, y1]
        })

        # Vẽ Bounding Box và ID lên ảnh để trực quan hóa
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2) # Màu xanh lá
        cv2.putText(image, str(i), (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Màu xanh dương

    # 5. Tính toán quan hệ không gian (is_above)
    # Logic: Node A nằm trên Node B nếu Node A kết thúc trước khi Node B bắt đầu theo chiều dọc
    relations = []
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i == j: continue
            
            a_x0, a_y0, a_x1, a_y1 = node_a["bbox"]
            b_x0, b_y0, b_x1, b_y1 = node_b["bbox"]

            # Điều kiện TRÊN - DƯỚI: A_y1 < B_y0 và có gióng hàng theo chiều dọc
            if a_y1 < b_y0 and (a_x0 < b_x1 and a_x1 > b_x0):
                # Ngưỡng khoảng cách gần (ví dụ 100 pixel trên ảnh lớn)
                if abs(b_y0 - a_y1) < 100: 
                    relations.append({
                        "source": node_a["id"],
                        "target": node_b["id"],
                        "rel": "is_above"
                    })

    # 6. Lưu ảnh trực quan hóa
    cv2.imwrite(output_viz_path, image)
    print(f"Đã lưu ảnh trực quan hóa tại: {output_viz_path}")

    # 7. Trả về cấu trúc đồ thị
    return {"nodes": nodes, "relations": relations}

# === CHẠY CHƯƠNG TRÌNH ===
# Hãy bỏ một file ảnh trang tài liệu (ví dụ: invoice.jpg) vào cùng thư mục
input_image = 'image_cua_ban.png' 

if os.path.exists(input_image):
    graph_data = process_image_to_graph(input_image)
    
    if graph_data:
        # Lưu đồ thị ra JSON để làm input cho LLM sinh câu hỏi
        with open("image_graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=4)
        
        print(f"Đã tạo đồ thị với {len(graph_data['nodes'])} nodes và {len(graph_data['relations'])} quan hệ.")
        print("Ví dụ quan hệ đầu tiên:", graph_data['relations'][0] if graph_data['relations'] else "Không tìm thấy quan hệ")
else:
    print(f"Vui lòng chuẩn bị file ảnh '{input_image}'")