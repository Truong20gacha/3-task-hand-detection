import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mlp_gesture_recognition.keras")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Gesture labels
gesture_labels = ['A', 'B', 'C', 'D']  # Update based on your labels

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks and flatten into a list
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)  # X coordinates
                landmarks.append(lm.y)  # Y coordinates
                # Skip Z if your model only uses X and Y (total 42 features)

            # Ensure landmarks match model input size
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks = landmarks / np.max(landmarks)  # Normalize to [0, 1]

            # Check if input shape matches model's expected shape
            if landmarks.shape[1] != model.input_shape[1]:
                print(f"Invalid input shape: {landmarks.shape}, expected: {model.input_shape[1]}")
                continue

            # Predict gesture
            prediction = model.predict(landmarks)
            gesture_index = np.argmax(prediction)
            gesture_name = gesture_labels[gesture_index]

            # Display the gesture on the video frame
            cv2.putText(frame, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
