# DocumentEdgeDetection(docgedect)
**DocumentEdgeDetection**은 이미지 속에서 문서를 자동으로 인식하여 문서의 테투리를 찾아내고, 인식된 문서를 스캔 이미지로 변형합니다.

### 주요 기능
- Preprocess - 이미지을 문서가 돋보이게 전처리합니다.
- Detect - 전처리한 이미지에서 Edge를 검출해 문서를 인식합니다.
- Reshape - 인식한 이미지을 원근변환을 통해 스캔 이미지로 변형합니다.

## 사용법
### 1. 의존 패키지 설치
```python
# Install dependency
pip install -r requirements.txt
```

### 2. 이미지 전처리
```python
# Image Preprocess
import docgedect as dd

image = cv2.imread('sample.jpg')
preprocessed = dd.preprocess.preprocess_image(image)
```

### 3. 문서 윤곽 검출
```python
# 코드 추가 예정
```

## 프로젝트 참여  
Email: **<neuroner0425@gmail.com>**