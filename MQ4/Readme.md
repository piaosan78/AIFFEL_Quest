---
## 코딩 진행
### 주요 목표 : 일단 결과를 만들어낸다.
* 아무리 나의 실력 상승을 위한거지만 일단 코딩의 결과를 내는 것이 먼저라고 생각하고 진행했습니다*

---
## 기본 틀
### 내용물 : 메인퀘스트 노드5장의 코드를 끌어와서 시간 절약
* 혼자 스스로 이해도 없이 진행하는 것은 무리라고 판단하여, 일단 가져올 수 있는 자료와 만들고 싶은 형식을 머리 속에서 꺼내서 GPT를 통한 resnet 개발을 진행했습니다.

 

```
# 배치 정규화 넣은 상태

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, MaxPooling2D

# First convolution layer
def residual_block(x, filters, kernel_size=3, stride=1):
    # Shortcut
    shortcut = x

    
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution layer
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Adjusting the shortcut for addition if needed
    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Adding the shortcut to the output
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x

def build_resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ResNet-18 모델 생성
resnet18_model = build_resnet18((180, 180, 3), 1)
resnet18_model.summary()
```

Loss: 0.10101824253797531,


Accuracy: 0.9649266004562378,


Precision: 0.9664268493652344,


Recall: 0.9817296266555786

![image](https://github.com/piaosan78/AIFFEL_Quest/assets/87675111/51f48693-0d7f-4aca-ab87-6b69de0a9c01)

### 정밀도는 정확도, 재현율, 손실 함수에 비해 낮지만 그래도 0.9인 상태라 만족스러운 결과입니다.

---
## kpt 회고
keep : 일단 약은 매일 복용하자. 우울해져서 무기력해지는 길은 피해야한다. 

prblum : 하지만 덕분에 코드 이해가 안되거나 지문이 이해가 안된다.

try : 저녁에 코드를 안 친다면 차라리 책을 1시간 정도 읽는 것을 해야한다.
