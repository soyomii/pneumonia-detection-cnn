import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# 🟢 경로 수정된 부분 (현재 폴더 기준으로 바로 접근)
img_dir = './unlabeled'
img_size = (150, 150)

# 이미지 불러오기
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpeg')]
x_data = []

for fname in img_files:
    img_path = os.path.join(img_dir, fname)
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    x_data.append(img_array)

x_data = np.array(x_data)

# 오토인코더 모델 정의
autoencoder = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(150,150,3)),
    MaxPooling2D((2,2), padding='same'),
    Conv2D(16, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2), padding='same'),
    Conv2D(16, (3,3), activation='relu', padding='same'),
    UpSampling2D((2,2)),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    UpSampling2D((2,2)),
    Conv2D(3, (3,3), activation='sigmoid', padding='same')
])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_data, x_data, epochs=10, batch_size=32, shuffle=True)

# 복원 이미지 확인
decoded_imgs = autoencoder.predict(x_data)
decoded_imgs = decoded_imgs.reshape(-1, img_size[0], img_size[1], 3)

plt.figure(figsize=(10, 4))
for i in range(5):
    # 원본
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_data[i].reshape(img_size[0], img_size[1], 3))
    plt.title("Original")
    plt.axis("off")

    # 복원된 이미지
    ax = plt.subplot(2, 5, i + 6)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
