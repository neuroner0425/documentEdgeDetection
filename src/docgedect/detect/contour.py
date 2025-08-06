import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

def smooth_contour(contour, image_shape, kernel_size=(8, 8), iterations=1):
    """
    contour를 마스크화한 후 morphological closing으로 smoothing

    Args:
        contour: 원본 contour
        image_shape: (height, width) of original image
        kernel_size: closing kernel 크기
        iterations: 연산 반복 횟수

    Returns:
        매끄럽게 다듬어진 contour (리스트)
    """
    # 빈 마스크 생성
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Closing 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # 새 contour 추출
    new_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if new_contours:
        # 가장 큰 contour 반환
        return max(new_contours, key=cv2.contourArea)
    else:
        return contour  # fallback


def find_document_contour(original_image: np.ndarray, edged: np.ndarray, save_dir: str = None, show_all: bool = False):
    """
    엣지 이미지에서 작은 노이즈 컨투어를 마스킹하여 제거한 뒤,
    문서(종이, 명함 등)로 추정되는 큰 사각형 영역의 윤곽선을 검출한다.
    다양한 파라미터 조정을 통해 자동 문서 추출 파이프라인에서 활용할 수 있다.

    Args:
        original_image (np.ndarray): 원본 컬러(BGR) 이미지.
        edged (np.ndarray): 엣지(이진) 이미지. 일반적으로 Canny, Morphology 등 후처리된 결과.
        save_dir (str, optional): 결과 시각화 이미지를 저장할 경로. 미지정 시 저장하지 않음.
        show_all (bool, optional): 중간 처리 결과 및 디버그 이미지를 OpenCV 창으로 표시할지 여부.

    Returns:
        screen_cnt (np.ndarray or None): 4점 근사화된 문서 외곽 컨투어(n,1,2) 또는 None.
        best_contour (np.ndarray or None): 내부적으로 smoothing 및 filtering을 거친 가장 큰 문서 컨투어. 
        contours (list): edged에서 검출된 상위 N개의 원본 컨투어 리스트.

    Example:
        cnt, best, all_contours = find_document_contour(img, canny_img)
    """
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = original_image.shape[:2]
    area_thresh = (min(h, w) * 0.1) ** 2

    # 작은 contour 마스크화
    small_mask = np.zeros_like(edged)
    for cnt in contours:
        if cv2.contourArea(cnt) < area_thresh:
            cv2.drawContours(small_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/small_mask.jpg', small_mask)
    if show_all:
        cv2.imshow('Small Mask', small_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 큰 contour 후보만 상위 8개
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    best_contour, screen_cnt, contour_area, contour_solidity = None, None, None, None

    for c in contours:
        doc_mask = np.zeros_like(edged)
        cv2.drawContours(doc_mask, [c], -1, 255, thickness=cv2.FILLED)
        doc_mask_clean = cv2.subtract(doc_mask, small_mask)
        clean_cnts, _ = cv2.findContours(doc_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not clean_cnts:
            continue
        cleaned = max(clean_cnts, key=cv2.contourArea)
        smoothed = smooth_contour(cleaned, (h, w))
        area = cv2.contourArea(smoothed)
        hull = cv2.convexHull(smoothed)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)

        if show_all:
            vis = original_image.copy()
            cv2.drawContours(vis, [c], -1, (0,255,0), 2)
            cv2.drawContours(vis, [approx], -1, (0,0,255), 2)
            cv2.drawContours(vis, [smoothed], -1, (255,0,0), 2)
            text = f"solidity: {solidity:.2f} / {len(approx)} / {int(area)}"
            cv2.putText(vis, text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.imshow('Contour Debug', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if len(approx) == 4 and solidity >= 0.9:
            if best_contour is None:
                best_contour = smoothed
                screen_cnt = approx
                contour_area = area
                contour_solidity = solidity
            elif area >= contour_area * 0.95 and solidity > contour_solidity:
                    best_contour = smoothed
                    screen_cnt = approx
                    contour_solidity = solidity

    if screen_cnt is None:
        return None, None, contours
    if show_all:
        cv2.imshow('Best Contour', cv2.drawContours(original_image.copy(), [best_contour], -1, (255, 0, 0), 2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_dir:
        img_draw = cv2.drawContours(original_image.copy(), [screen_cnt], -1, (0,255,0), 2)
        img_draw = cv2.drawContours(img_draw, [best_contour], -1, (255,0,0), 2)
        cv2.imwrite(f'{save_dir}/contour.jpg', img_draw)

    return screen_cnt, best_contour, contours
