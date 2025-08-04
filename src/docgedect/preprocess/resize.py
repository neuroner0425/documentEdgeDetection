import cv2
import numpy as np

def resize_image(image: np.ndarray, max_size: float = 1000.0) -> np.ndarray:
    """
    이미지 크기 조정.

    Args:
        image: 입력 이미지
        max_size: 최대 한 변 크기

    Returns:
        크기 조정된 이미지
    """
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        ratio = max_size / max(height, width)
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    return image
