import os
import cv2
import mediapipe as mp
import pandas as pd

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Đường dẫn
input_folder = r"E:/pythonProject/pythonProject1/dataset_word3"
output_folder = "E:/pythonProject/pythonProject1/processed_dataset"
os.makedirs(output_folder, exist_ok=True)

# Gán nhãn cho các thư mục
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Danh sách lưu thông tin ảnh và nhãn
data_info = []


def preprocess_image(image_path, label):
    """
    Tiền xử lý ảnh:
    - Resize ảnh về kích thước 640x480.
    - Tăng độ sáng và độ tương phản.
    - Dùng MediaPipe để vẽ khung tay lên ảnh (nếu có).
    """
    print(f"Đang xử lý ảnh: {image_path}")

    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return None, None, None  # Ảnh không hợp lệ

    print(f"  Đọc ảnh thành công: Kích thước {img.shape}, Pixel trung bình: {img.mean()}")

    # Resize ảnh về 640x480
    img_resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    print(f"  Kích thước sau resize: {img_resized.shape}, Pixel trung bình: {img_resized.mean()}")

    # Tăng độ sáng và độ tương phản
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=40)
    print(f"  Tăng sáng hoàn tất, Pixel trung bình sau khi tăng sáng: {img_rgb.mean()}")

    # Kiểm tra bàn tay bằng MediaPipe
    result = hands.process(img_rgb)

    landmarks = []  # Lưu tọa độ nếu có
    if result.multi_hand_landmarks:
        print("  Phát hiện bàn tay!")
        # Vẽ khung tay nếu phát hiện
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_resized,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                landmarks.append((x, y))
    else:
        print("  Không phát hiện bàn tay.")

    return img_resized, label, landmarks


# Lặp qua các thư mục và xử lý ảnh
for folder in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder)
    if folder not in label_mapping:
        print(f"Thư mục không hợp lệ, bỏ qua: {folder}")
        continue  # Bỏ qua thư mục không hợp lệ

    label = label_mapping[folder]
    output_subfolder = os.path.join(output_folder, folder)
    os.makedirs(output_subfolder, exist_ok=True)

    print(f"Đang xử lý thư mục: {folder}")
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        # Tiền xử lý ảnh
        processed_image, processed_label, landmarks = preprocess_image(image_path, label)

        # Lưu ảnh đã xử lý (có hoặc không có khung tay)
        if processed_image is not None:
            output_image_path = os.path.join(output_subfolder, image_file)
            cv2.imwrite(output_image_path, processed_image)

            # Tạo dữ liệu hàng cho CSV
            row = {
                'file_name': os.path.join(folder, image_file),
                'label': processed_label
            }
            # Thêm tọa độ landmarks (x, y) nếu có
            if landmarks:
                for i, (x, y) in enumerate(landmarks):
                    row[f'x{i}'] = x
                    row[f'y{i}'] = y

            # Lưu thông tin vào danh sách
            data_info.append(row)

# Lưu thông tin nhãn và tọa độ vào CSV
df = pd.DataFrame(data_info)
df.to_csv(os.path.join(output_folder, "labels_with_landmarks.csv"), index=False)
print("Quy trình tiền xử lý hoàn tất!")
