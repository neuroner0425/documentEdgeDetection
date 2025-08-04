from .preprocess import (
    preprocess_image, opencv2pil, resize_image,
    reduce_lighting, adjust_contrast_exposure,
    adjust_black_point, adjust_highlight
)

import cv2
import numpy as np

def find_document_contour(original_image: np.ndarray, edged: np.ndarray):
    """
    윤곽선을 찾고, 4개 꼭짓점의 문서 윤곽을 반환

    Args:
        original_image: 원본 이미지 (BGR)
        edged: 엣지 이미지 (binary)

    Returns:
        문서 영역 contour (4점) 또는 None
    """
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    print(f"STEP 2: 윤곽선 찾기 시작, 총 {len(contours)}개 발견.")

    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        print("문서의 윤곽선을 찾지 못했습니다.")
        return None
    else:
        print("STEP 2: 윤곽선 찾기 완료")
        cv2.imwrite('./out/contour_image.jpg', cv2.drawContours(original_image.copy(), [screen_cnt], -1, (0, 255, 0), 2))
        return screen_cnt
