from flask import Flask, render_template, request, render_template_string
import cv2
import torch
from ultralytics import YOLO
import os

app = Flask(__name__, static_folder="static")

# Load mô hình YOLOv8
model = YOLO("best.pt")  # Thay bằng đường dẫn mô hình của bạn

# Danh sách mô tả đối tượng
OBJECT_INFO = {
    "dambut": "Hoa dâm bụt (Hibiscus rosa-sinensis) là loài hoa thân gỗ nhỏ, có nguồn gốc từ khu vực nhiệt đới châu Á. Hoa lớn, sặc sỡ, thường có màu đỏ, vàng, cam hoặc hồng. Cây dâm bụt thường được trồng làm hàng rào, trang trí sân vườn hoặc công viên. Trong y học cổ truyền, hoa và lá dâm bụt được dùng để chữa ho, thanh nhiệt, làm mát gan và hỗ trợ tiêu hóa. Ngoài ra, dâm bụt còn được dùng để pha trà thảo mộc, giúp hạ huyết áp và làm dịu cơ thể.",
    "duongsi": "Dương xỉ là nhóm thực vật cổ xưa, sinh trưởng tốt trong môi trường ẩm ướt, râm mát. Lá dương xỉ thường có dạng lông chim, xanh tươi quanh năm, tạo cảm giác mát mẻ, thư giãn. Cây thường được trồng trong nhà, văn phòng, ban công để thanh lọc không khí và giảm căng thẳng. Dương xỉ không chỉ có giá trị làm cảnh mà còn đóng vai trò trong việc cải tạo đất và chống xói mòn ở môi trường tự nhiên.",
    "hong": "Hoa hồng (Rosa spp.) là một trong những loài hoa nổi tiếng nhất thế giới, biểu tượng của tình yêu, sự lãng mạn và cái đẹp. Có hơn 100 loài hoa hồng với màu sắc đa dạng như đỏ, trắng, hồng, vàng... Mỗi màu mang một ý nghĩa riêng: đỏ tượng trưng cho tình yêu mãnh liệt, trắng cho sự thuần khiết, vàng cho tình bạn. Hoa hồng được trồng để trang trí vườn, cắm hoa, làm nước hoa và mỹ phẩm. Ngoài ra, cánh hoa hồng còn được dùng làm trà, có tác dụng làm đẹp da và giảm stress.",
    "kimtien": "Cây kim tiền (Zamioculcas zamiifolia) là loại cây cảnh phong thủy được ưa chuộng trong trang trí nội thất và văn phòng. Cây có thân mập, lá xanh bóng, mọc đối xứng đẹp mắt. Theo quan niệm phong thủy, kim tiền tượng trưng cho sự phát tài, phát lộc, mang lại may mắn và thịnh vượng cho gia chủ. Cây rất dễ sống, chịu hạn tốt và có thể trồng trong môi trường thiếu sáng, thích hợp cho người bận rộn.",
    "senda": "Sen đá (Succulent) là tên gọi chung cho các loài cây mọng nước có hình dáng nhỏ gọn, lá dày, trữ nước. Sen đá có hàng trăm loại với hình dạng và màu sắc phong phú như xanh, tím, hồng, viền đỏ... Cây thích hợp trồng trong chậu nhỏ để bàn, trang trí không gian sống và làm việc. Sen đá tượng trưng cho sự bền bỉ, kiên cường và tình cảm lâu dài. Đây cũng là món quà tặng phổ biến trong các dịp đặc biệt nhờ vẻ đẹp tinh tế và ý nghĩa sâu sắc."
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = "static/uploads/original.jpg"
            file.save(img_path)

            # Chạy YOLOv8 nhận diện
            results = model(img_path)

            # Đọc ảnh
            img = cv2.imread(img_path)

            # Duyệt qua tất cả bounding boxes và vẽ lên ảnh
            detections = results[0].boxes.data.cpu().numpy()
            all_labels = []  # Danh sách kết quả nhận diện
            for box in detections:
                x1, y1, x2, y2, conf, cls_id = box
                label = model.names[int(cls_id)]
                confidence = round(float(conf) * 100, 2)  # Chuyển về %

                # Thêm thông tin vào danh sách
                all_labels.append((label, confidence))

                # Vẽ bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.putText(img, f"{label} {confidence}%", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            detected_img_path = "static/uploads/detected.jpg"
            cv2.imwrite(detected_img_path, img)

            # Lấy mô tả đối tượng đầu tiên (nếu có)
            if all_labels:
                main_label, main_conf = all_labels[0]
                description = OBJECT_INFO.get(main_label, "Không có thông tin.")
            else:
                main_label, main_conf, description = "Không có đối tượng", 0, "Không có mô tả."

            return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Website nhận diện hình ảnh</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            padding: 30px;
            max-width: 1000px;
            width: 100%;
            animation: fadeIn 0.5s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 2rem;
            font-weight: 600;
        }

        .upload-form {
            text-align: center;
            margin-bottom: 30px;
        }

        .upload-form input[type="file"] {
            display: none;
        }

        .upload-label {
            background: #3498db;
            color: #fff;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-block;
            font-size: 1rem;
        }

        .upload-label:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }

        .result-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }

        .image-box img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .image-box img:hover {
            transform: scale(1.02);
        }

        .info-box {
            background: linear-gradient(135deg, #f9f9f9 0%, #e8f0fe 100%);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }

        .info-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(to right, #ff6f61, #ffb347);
        }

        .info-box h2 {
            color: #e74c3c;
            font-size: 1.5rem;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .info-box p {
            color: #34495e;
            line-height: 1.6;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.8);
            padding: 15px;
            border-left: 4px solid #ff6f61;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease, background 0.3s ease;
        }

        .info-box p:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 1);
        }

        .info-box p::first-letter {
            font-size: 1.5rem;
            color: #ff6f61;
            font-weight: bold;
            float: left;
            margin-right: 5px;
            line-height: 1;
        }

        .labels-list {
            list-style: none;
            margin-top: 15px;
        }

        .labels-list li {
            background: #ecf0f1;
            padding: 8px 15px;
            border-radius: 20px;
            margin-bottom: 10px;
            color: #7f8c8d;
            font-size: 0.9rem;
            transition: background 0.3s ease;
        }

        .labels-list li:hover {
            background: #dfe6e9;
        }

        @media (max-width: 768px) {
            .result-section {
                grid-template-columns: 1fr;
            }
            .container {
                padding: 20px;
            }
            h1 {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Website nhận diện hình ảnh</h1>
        
        <form method="POST" enctype="multipart/form-data" class="upload-form">
            <label for="image" class="upload-label">Chọn ảnh để nhận diện</label>
            <input type="file" id="image" name="image" accept="image/*" onchange="this.form.submit()">
        </form>

        {% if original and detected %}
        <div class="result-section">
            <div class="image-box">
                <h3>Ảnh gốc</h3>
                <img src="{{ original }}" alt="Original Image">
            </div>
            <div class="image-box">
                <h3>Ảnh nhận diện</h3>
                <img src="{{ detected }}" alt="Detected Image">
            </div>
            <div class="info-box">
                <h2>{{ main_label }} ({{ main_conf }}%)</h2>
                <p>{{ description }}</p>
                {% if labels|length > 1 %}
                <h3>Các đối tượng khác:</h3>
                <ul class="labels-list">
                    {% for label, conf in labels[1:] %}
                    <li>{{ label }} ({{ conf }}%)</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
            ''', 
            original=img_path, 
            detected=detected_img_path, 
            labels=all_labels,
            main_label=main_label,
            main_conf=main_conf,
            description=description)

    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Website nhận diện hình ảnh</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            padding: 30px;
            max-width: 1000px;
            width: 100%;
            animation: fadeIn 0.5s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 2rem;
            font-weight: 600;
        }

        .upload-form {
            text-align: center;
            margin-bottom: 30px;
        }

        .upload-form input[type="file"] {
            display: none;
        }

        .upload-label {
            background: #3498db;
            color: #fff;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-block;
            font-size: 1rem;
        }

        .upload-label:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Website nhận diện hình ảnh </h1>
        
        <form method="POST" enctype="multipart/form-data" class="upload-form">
            <label for="image" class="upload-label">Chọn ảnh để nhận diện</label>
            <input type="file" id="image" name="image" accept="image/*" onchange="this.form.submit()">
        </form>
    </div>
</body>
</html>
    ''')

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)