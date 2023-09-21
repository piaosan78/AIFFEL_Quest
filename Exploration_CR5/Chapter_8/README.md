# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 박영준
- 리뷰어 : 박근수


# PRT(PeerReviewTemplate) 
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - DCGAN 모델을 활용하여 새로운 이미지가 생성되는 것을 확인함
    - 이미지 시각화 및 결과 확인하였음
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 전체적으로 코드에 주석이 부족하지만 필요한 부분은 달려있어 이해하기 어렵지 않았다.
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 이미지 업스케일링, 가우시안 노이즈 추가, 레이블 스무딩 적용하여 새로운 시도 및 추가 실험을 통해
      Real Accuracy: 55.0, Fake Accuracy: 98.828125 달성
      
    ```python]
    #이미지 업스케일링
    upscaled_size = (96, 96)
    upscaled_images = tf.image.resize(train_images, upscaled_size)
    ```
    ```python
    # Add Gaussian noise to real and generated images
        images_with_noise = add_gaussian_noise_to_input(images)
        generated_images_with_noise = add_gaussian_noise_to_input(generated_images)

        real_output = discriminator(images_with_noise, training=True)
        fake_output = discriminator(generated_images_with_noise, training=True)

        # 레이블 스무딩 적용
        real_labels_smoothed = tf.ones_like(real_output) * 0.9
        fake_labels = tf.zeros_like(fake_output)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_labels_smoothed, real_output, fake_labels, fake_output)
    ```
        
- [X]  **4. 회고를 잘 작성했나요?**
    - 작성된 회고는 없었지만 대화를 통해 시간이 오래 걸리는 모델임에도 여러 시도를 해보았음
    - 시간 관계상 에폭을 더 늘려 학습해보지 못해 추후 에폭수 증가 및 추가 개선방안 적용하여 시도예정

- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
        - 코드가 대체로 파이썬 스타일 가이드를 따르고 있으며 간결하고 효율적으로 작성되어 있음
    - 부가적인 개선사항
1. 주석(comment)에 대한 개선:
주석은 코드의 가독성을 높이는 데 도움이 되지만, 주석이 코드보다 길어지지 않도록 주의해야 합니다. 몇몇 주석은 과도하게 자세한 내용을 설명하고 있으며, 줄임말이나 간결한 설명으로 대체할 수 있습니다. 주석의 가독성을 높이려면 다음과 같이 개선할 수 있습니다:

2. 함수와 변수의 명명 개선:
함수와 변수의 이름은 가독성을 높이기 위해 더 명확하게 지정할 수 있습니다. 예를 들어, make_generator_model 및 make_discriminator_model 대신 create_generator_model 및 create_discriminator_model과 같이 함수 이름을 바꿀 수 있습니다.

3. 매직 넘버(Magic Number)의 사용 줄이기:
매직 넘버는 코드의 가독성을 낮출 수 있으므로, 숫자 상수에 의미 있는 이름을 부여하면 코드를 이해하기 쉬워집니다. 예를 들어, BATCH_SIZE = 256와 같이 상수를 사용하여 의미를 명확히 할 수 있습니다.

4. 학습률(Learning Rate)의 조정:
학습률은 모델 학습에 중요한 요소이며, 최적의 학습률을 찾는 것이 중요합니다. 학습률을 변경하며 실험하여 최적의 학습률을 찾아보세요.

5. GAN 모델의 복잡도 조절:
GAN 모델의 복잡도를 조절하여 원하는 성능과 생성된 이미지 품질을 얻을 수 있습니다. 더 복잡한 모델을 사용하거나 더 깊은 신경망 층을 추가하여 결과를 향상시킬 수 있습니다.

6. 학습 시간 측정 개선:
time 모듈을 사용하여 훈련 시간을 측정하고 있지만, 더 정확한 시간 측정을 위해 훈련 시작 및 종료 시점을 더 정확하게 측정하는 방법을 고려해 볼 수 있습니다.

7. 불필요한 import 정리:
코드의 가독성을 높이기 위해 필요하지 않은 import 문을 정리할 수 있습니다.
  
# 참고 링크 및 코드 개선
- 아래 개선 코드 참고
```python
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10

# 데이터 로딩 및 전처리
(train_images, _), (_, _) = cifar10.load_data()
train_images = (train_images - 127.5) / 127.5  # 이미지를 [-1, 1]로 정규화

# 이미지 업스케일링 설정
upscaled_size = (96, 96)
upscaled_images = tf.image.resize(train_images, upscaled_size)

# 업스케일링된 이미지 시각화 함수
def plot_images(images, n):
    fig, axes = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        image = (images[i] + 1) / 2.0  # [-1, 1] 범위를 [0, 1]로 변경
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.show()

# 데이터셋 설정
BUFFER_SIZE = train_images.shape[0]  # 데이터셋 버퍼 크기 설정
BATCH_SIZE = 256  # 배치 크기 설정
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

NOISE_DIM = 100  # 생성자 입력으로 사용할 노이즈 벡터의 차원 설정

# 생성자와 판별자 모델 생성 함수
def create_generator_model():
    # 생성자 모델 정의
    model = Sequential([
        Dense(512, use_bias=False, input_shape=(NOISE_DIM,)),  # 입력 노이즈 벡터를 받는 레이어
        BatchNormalization(),
        LeakyReLU(),

        Dense(8*8*256, use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Reshape((8, 8, 256)),

        Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')  # 출력 레이어
    ])
    return model

def create_discriminator_model():
    # 판별자 모델 정의
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),  # 입력 이미지를 받는 합성곱 레이어
        LeakyReLU(),
        Dropout(0.3),  # 드롭아웃 레이어 (과적합 방지를 위해 랜덤하게 뉴런을 비활성화)

        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),  # 다차원 텐서를 평평한 형태로 펼치는 레이어
        Dense(1)  # 판별 결과를 출력하는 레이어
    ])
    return model

# 나머지 코드는 그대로 유지

```
