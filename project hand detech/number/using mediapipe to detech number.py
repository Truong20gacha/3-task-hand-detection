import cv2
import mediapipe as mp

# Khởi tạo MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Mở camera
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """Xác định số ngón tay giơ lên để nhận diện số từ 0-5."""
    fingers = [4, 8, 12, 16, 20]  # Các đầu ngón tay
    count = 0
    thumb_up = False

    wrist_y = hand_landmarks.landmark[0].y  # Cổ tay (dùng để kiểm tra nắm tay)
    distances = [abs(hand_landmarks.landmark[f].y - wrist_y) for f in fingers]

    # Nếu tất cả khoảng cách đều rất nhỏ (tay nắm lại) -> số 0
    if all(d < 0.05 for d in distances):
        return 0

    # Kiểm tra các ngón tay mở
    for i in range(1, 5):  # Kiểm tra ngón trỏ đến ngón út
        if hand_landmarks.landmark[fingers[i]].y < hand_landmarks.landmark[fingers[i] - 2].y:
            count += 1  # Nếu đầu ngón tay cao hơn khớp dưới, thì ngón đó giơ lên

    # Kiểm tra ngón cái
    thumb_x = hand_landmarks.landmark[4].x
    thumb_mcp_x = hand_landmarks.landmark[3].x
    thumb_y = hand_landmarks.landmark[4].y
    thumb_mcp_y = hand_landmarks.landmark[2].y
    wrist_x = hand_landmarks.landmark[0].x

    if (thumb_x > thumb_mcp_x and wrist_x < thumb_x) or (thumb_x < thumb_mcp_x and wrist_x > thumb_x):
        if thumb_y < thumb_mcp_y:  # Đảm bảo ngón cái mở đúng hướng
            thumb_up = True

    return count + (1 if thumb_up else 0)

def detect_hand_orientation(hand_landmarks, hand_label):
    """Xác định mặt bàn tay (Mặt trước hay Mặt Sau) với cùng quy tắc cho cả hai tay."""
    wrist_x = hand_landmarks.landmark[0].x  # Cổ tay
    thumb_x = hand_landmarks.landmark[4].x

    if hand_label == "Left":
        return "front" if thumb_x > wrist_x else "back"
    else:  # Tay phải
        return "front" if thumb_x < wrist_x else "back"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_numbers = []  # Lưu số của mỗi tay
    orientations = []  # Lưu mặt trước / mặt sau của mỗi tay

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[hand_index].classification[0].label  # "Left" hoặc "Right"
            fingers_count = count_fingers(hand_landmarks)
            hand_orientation = detect_hand_orientation(hand_landmarks, hand_label)

            # Nếu là mặt sau, cộng thêm 5 vào số nhận diện
            if hand_orientation == "back":
                fingers_count += 5

            detected_numbers.append(fingers_count)
            orientations.append(hand_orientation)

            # Vẽ keypoints lên ảnh
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Xử lý logic cộng số
    total_number = sum(detected_numbers)

    # Nếu cả hai tay đều giơ hết 5 ngón và đều ở mặt sau, gán giá trị 20 thay vì 10+10
    if len(detected_numbers) == 2 and detected_numbers == [10, 10] and orientations == ["back", "back"]:
        total_number = 20

    # Hiển thị số nhận diện
    cv2.putText(frame, f"Number: {total_number}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
