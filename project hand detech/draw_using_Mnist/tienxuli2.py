import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os
import matplotlib.pyplot as plt

# Đường dẫn lưu file dataset
dataset_path = r"E:\pythonProject\pythonProject1\mnist_dataset.npz"

# 1) Kiểm tra nếu file dataset tồn tại thì tải dữ liệu từ file .npz, nếu không thì tải từ TensorFlow
if os.path.exists(dataset_path):
    data = np.load(dataset_path)
    train_images, train_labels = data['train_images'], data['train_labels']
    test_images, test_labels = data['test_images'], data['test_labels']
else:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    np.savez_compressed(dataset_path, train_images=train_images, train_labels=train_labels,
                        test_images=test_images, test_labels=test_labels)

# 2) Chuẩn hóa dữ liệu (chuyển về [0,1]) và thêm chiều kênh (28,28) -> (28,28,1)
train_images = train_images.astype('float32') / 255.0
test_images  = test_images.astype('float32') / 255.0
train_images = np.expand_dims(train_images, axis=-1)
test_images  = np.expand_dims(test_images, axis=-1)

# 3) Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 4) Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5) Huấn luyện mô hình với validation_split=0.1 và epochs=20
history = model.fit(
    train_images, train_labels,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 6) Trực quan hoá quá trình huấn luyện
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 7) Đánh giá mô hình trên tập test (test set chỉ dùng để đánh giá cuối cùng)
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 8) Lưu mô hình sau khi huấn luyện
model.save(r"E:\pythonProject\pythonProject1\mnist_model_3.h5")
print("Mô hình đã được lưu tại: E:\\pythonProject\\pythonProject1\\mnist_model_3.h5")
