<<<<<<< HEAD
#  Pneumonia Detection CNN

AI를 활용한 흉부 X-ray 영상 기반 폐렴 진단 모델 구현 프로젝트입니다.  
지도학습(CNN 기반 이진 분류 모델)을 통해 **정상 vs 폐렴** 이미지를 구분하는 모델을 만들었습니다.  
이 프로젝트는 이후 진행된 **XAI 기반 비지도학습 프로젝트**의 선행 실험으로 진행되었습니다.  
(→ 이미 완료된 [XAI 프로젝트 바로가기](https://github.com/soyomii/pneumonia-anomaly-xai))



---



## 🚧 프로젝트 개요

- **기획 배경**:  
  폐렴 진단을 위한 X-ray 데이터셋을 CNN으로 분류해보고,  
  이후 라벨이 없는 경우에는 어떤 방식으로 진단 가능한지 실험하고자 기획됨.

- **목표 설정**:  
  지도학습 기반 이진 분류 모델을 만들고,  
  성능 확인 후 추후 FastAPI 서비스 및 비지도 학습(XAI)으로 확장



---



## 🛠 기술 스택

- **Language**: Python  
- **Framework/Library**: TensorFlow, Keras, Matplotlib, NumPy  
- **Tool**: VSCode, Git, GitHub  
- **Dataset**: Chest X-ray 이미지 (Normal / Pneumonia)



---



## ✅ 주요 기능

- CNN 기반 이진 분류 모델 설계
- 학습/검증 정확도 및 손실 시각화
- 실제 X-ray 이미지 입력 시 폐렴 여부 예측 및 시각적 라벨링 출력



---



## 📊 모델 성능

- **Train Accuracy**: 약 98%  
- **Validation Accuracy**: 약 87.5%  
- 오버피팅 없이 일반화된 성능 확보



---



## 🧠 배운 점

- 딥러닝(CNN) 구조 설계 및 실전 적용
- 이미지 데이터 전처리 및 증강 경험
- 학습 이력 시각화 및 오차 분석
- 프로젝트 구조화 및 GitHub 포트폴리오화
- 라벨 유무에 따른 접근 방식(지도 vs 비지도) 기획 감각 향상



---



## 🚀 차후 계획

- ✅ **라벨 제거 후 비지도 기반 오토인코더 실험 완료**  
- ✅ **보고서 작성 완료**
- ⏸ **FastAPI를 통해 웹 예측 서비스 제공** (예정)  




---


## 📊 시각화 결과



모델의 학습 성능과 이상 탐지 결과를 시각화하였습니다.




### ✅ 학습 정확도 및 손실 시각화  
![학습 정확도 및 손실 시각화](https://github.com/soyomii/pneumonia-detection-cnn/blob/main/images/%ED%95%99%EC%8A%B5%EC%A0%95%ED%99%95%EB%8F%84%20%EB%B0%8F%20%EC%86%90%EC%8B%A4%20%EC%8B%9C%EA%B0%81%EB%8F%84.png?raw=true)



### 🧪 실제 예측 결과 예시  
![실제 예측 결과 예시](https://github.com/soyomii/pneumonia-detection-cnn/blob/main/images/%EC%8B%A4%EC%A0%9C%20%EC%98%88%EC%B8%A1%20%EA%B2%B0%EA%B3%BC%20%EC%98%88%EC%8B%9C.png?raw=true)



---



## 📘 용어 정리

| 용어 | 설명 |
|------|------|
| **CNN (Convolutional Neural Network)** | 이미지 분류에 특화된 딥러닝 구조로, 합성곱 연산을 통해 이미지의 특징을 추출함 |
| **Epoch** | 전체 데이터셋을 모델이 한 번 학습하는 주기 (ex: 3 epoch = 데이터셋 3번 반복 학습) |
| **Batch Size** | 모델이 한 번에 학습하는 데이터 수. 메모리 효율성과 학습 안정성에 영향을 줌 |
| **Accuracy** | 예측이 실제 값과 얼마나 맞았는지를 백분율로 표현한 지표 |
| **Loss** | 모델이 얼마나 틀렸는지를 수치화한 값. 낮을수록 좋은 모델 |
| **Binary Crossentropy** | 이진 분류 문제에서 사용하는 손실 함수. 예측이 정답과 얼마나 다른지를 계산함 |
| **Activation Function (ReLU, Sigmoid)** | 뉴런이 활성화되는 방식을 정함. ReLU는 음수를 0으로 만들고, Sigmoid는 값을 0~1로 압축함 |
| **Overfitting (과적합)** | 학습 데이터에는 성능이 좋지만, 새로운 데이터에는 성능이 떨어지는 현상 |
| **ImageDataGenerator** | 이미지 데이터를 자동으로 증강하거나 전처리해주는 Keras 도구 |



---



📁 `images/` 폴더에 학습 시각화 결과 포함 (ex: 정확도 그래프, 예측 결과 등)



---




📌 본 프로젝트는 개인 포트폴리오 목적이며, 의료 진단 도구로 사용되지 않습니다.
=======

