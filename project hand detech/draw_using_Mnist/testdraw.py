import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ===== 1) Chuẩn bị =====
# Tải mô hình MNIST đã huấn luyện
# co the thu voi mnist_model_improved.h5 nhu cc
# co the thu voi mnist_model_3.h5
model = load_model("mnist_model_3.h5")

# Khởi tạo Mediapipe cho việc nhận diện bàn tay
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Khởi tạo camera
cap = cv2.VideoCapture(0)
ret, frame_init = cap.read()
if not ret:
    print("Không thể truy cập camera.")
    cap.release()
    exit()

# Lật frame để có hiệu ứng gương (mirror)
frame_init = cv2.flip(frame_init, 1)
height, width = frame_init.shape[:2]

# Canvas đen để vẽ (1 kênh)
canvas = np.zeros((height, width), dtype=np.uint8)

# Lưu toạ độ ngón trỏ cũ (để vẽ nét)
prev_x, prev_y = None, None


# ===== 2) Hàm dự đoán chữ số từ canvas =====
def predict_digit_from_canvas(canvas_img, model):
    """
    1. Tìm bounding box quanh vùng có nét vẽ (pixel > 0).
    2. Cắt ROI, resize 28x28, chuẩn hóa.
    3. Đưa vào mô hình MNIST để dự đoán.
    """
    pts = cv2.findNonZero(canvas_img)
    if pts is None:
        return None  # Không có nét vẽ nào

    # Lấy bounding box (x,y,w,h) quanh vùng vẽ
    x, y, w, h = cv2.boundingRect(pts)
    roi = canvas_img[y:y + h, x:x + w]

    # Resize về 28x28
    roi = cv2.resize(roi, (28, 28))

    # MNIST chuẩn: chữ số trắng trên nền đen
    # Ở canvas ta đã vẽ nét trắng (255) trên nền đen (0), nên không cần đảo màu
    # Nếu bạn lỡ vẽ đen trên trắng, hãy dùng: roi = cv2.bitwise_not(roi)

    # Chuẩn hóa
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)  # (1, 28, 28)
    roi = np.expand_dims(roi, axis=-1)  # (1, 28, 28, 1)

    # Dự đoán
    pred = model.predict(roi)
    digit = np.argmax(pred[0])
    return digit


# ===== 3) Vòng lặp chính =====
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Vẽ 1 tay
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lật ngang để tạo hiệu ứng gương
        frame = cv2.flip(frame, 1)

        # Chuyển ảnh sang RGB (mediapipe yêu cầu)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Xử lý bằng mediapipe
        result = hands.process(rgb_frame)

        # Lấy kết quả các điểm (landmarks) của bàn tay
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                # (Nếu muốn vẽ khung xương bàn tay cho debug)

                # Lấy toạ độ ngón trỏ (index finger tip) - landmark số 8
                # Mediapipe đánh số:
                #   0 - wrist, 1 - thumbCMC, 4 - thumbTip, 8 - indexTip, ...
                index_tip = handLms.landmark[8]

                # Chuyển toạ độ chuẩn hoá [0..1] sang pixel
                cx = int(index_tip.x * width)
                cy = int(index_tip.y * height)

                # Vẽ 1 chấm xanh tại đầu ngón trỏ để debug
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Nếu đã có prev_x, prev_y thì vẽ 1 line trên canvas
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), 255, 5)

                # Cập nhật vị trí ngón trỏ cũ
                prev_x, prev_y = cx, cy
        else:
            # Không tìm thấy tay, reset prev_x, prev_y
            prev_x, prev_y = None, None

        # Hiển thị text hướng dẫn
        cv2.putText(frame, "Press 's' to predict, 'c' to clear, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị khung camera
        cv2.imshow("Camera", frame)
        # Hiển thị canvas
        cv2.imshow("Canvas", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Xoá canvas
            canvas = np.zeros((height, width), dtype=np.uint8)
            prev_x, prev_y = None, None
        elif key == ord('s'):
            # Dự đoán chữ số
            digit = predict_digit_from_canvas(canvas, model)
            if digit is not None:
                print("Dự đoán chữ số:", digit)
            else:
                print("Không tìm thấy nét vẽ!")

cap.release()
cv2.destroyAllWindows()
