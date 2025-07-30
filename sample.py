import os
import cv2
import numpy as np
import docgedect as ddect

img_paths = [
    'input.jpg',  # 여기에 처리할 이미지 경로를 추가하세요
]
img_paths = [f'./document_data/img/{f}' for f in os.listdir('./document_data/img')
             if os.path.isfile(os.path.join('./document_data/img', f)) and
             f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

# 1. 이미지 불러오기 및 전처리
def preprocess_image(img_path):
    os.makedirs('./out/edge', exist_ok=True)
    os.makedirs('./out/photo_edges', exist_ok=True)
    image = cv2.imread(img_path)
    # 원본 이미지의 복사본을 만들어 비율을 유지하며 리사이즈 (너무 크면 처리 속도가 느려짐)
    orig = image.copy()
    height, width = image.shape[:2]
    max_size = 1000.0  # 최대 크기 설정 (예: 1000 픽셀)
    if max(height, width) > max_size:
        ratio = max_size / max(height, width)
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./out/0.gray.jpg', gray)
    cv2.imwrite('./out/edge/0.gray.jpg', cv2.Canny(gray, 75, 200))

    # 아이폰 사진 기준 대비 50%, 노출 -100 적용 (gray)
    exposure_adjusted = ddect.adjust_contrast_exposure(gray, 1.5, exposure=-100)
    cv2.imwrite('./out/1.exposure_adjusted_image.jpg', exposure_adjusted)
    cv2.imwrite('./out/edge/1.exposure_adjusted_image.jpg', cv2.Canny(exposure_adjusted, 75, 200))

    # # 하이라이트 +100 적용 (gray)
    highlight_mask = exposure_adjusted > 200  # 밝은 영역만 선택
    highlight_adjusted = exposure_adjusted.copy()
    highlight_adjusted[highlight_mask] = np.clip(highlight_adjusted[highlight_mask] + 100, 0, 255)
    cv2.imwrite('./out/2.highlight_adjusted_image.jpg', highlight_adjusted)
    cv2.imwrite('./out/edge/2.highlight_adjusted_image.jpg', cv2.Canny(highlight_adjusted, 75, 200))

    black_point_adjusted = ddect.adjust_black_point(highlight_adjusted, threshold=80)
    cv2.imwrite('./out/4.black_point_adjusted_image.jpg', black_point_adjusted)
    cv2.imwrite('./out/edge/4.black_point_adjusted_image.jpg', cv2.Canny(black_point_adjusted, 75, 200))

    preprocessed_image = exposure_adjusted

    edged = cv2.Canny(preprocessed_image, 75, 150)
    cv2.imwrite('./out/edged_image.jpg', edged)

    print("STEP 1: 엣지 검출 완료")
    
    return image, preprocessed_image, edged

def keep_long_segments(edge_img, min_length=50):
    kernel = np.ones((10,10), np.uint8)
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel) 
    cv2.imwrite('./out/filtered_edges.jpg', closed)
    return closed

# 2. 윤곽선 찾기 및 필터링
def find_document_contour(original_image, edged):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 찾은 윤곽선들을 크기 순으로 정렬
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    
    print(f"STEP 2: 윤곽선 찾기 시작, 총 {len(contours)}개의 윤곽선이 발견되었습니다.")
    
    for c in contours:
    
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            copy_original_image= original_image.copy()
            cv2.imshow("Contour", copy_original_image)
            cv2.drawContours(copy_original_image, [c], -1, (0, 255, 0), 2)
            cv2.imshow("Contour", copy_original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    screenCnt = None

    # 윤곽선 루프를 돌며 꼭짓점이 4개인 것을 찾음
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("문서의 윤곽선을 찾지 못했습니다.")
        return None
    else:
        print("STEP 2: 윤곽선 찾기 완료")
        cv2.imwrite('./out/contour_image.jpg', cv2.drawContours(original_image.copy(), [screenCnt], -1, (0, 255, 0), 2))
        return screenCnt


if __name__ == "__main__":
    for img_path in img_paths:
        print(f"Processing {img_path}...")
        
        file_name, _ = os.path.splitext(os.path.basename(img_path))
        result_path = os.path.join('document_data', 'result', file_name)
        os.makedirs(result_path, exist_ok=True)

        # 1. 이미지 불러오기 및 전처리
        image, preprocess_img, edged = preprocess_image(img_path)
        cv2.imwrite(os.path.join(result_path, 'preprocess.jpg'), preprocess_img)
        cv2.imwrite(os.path.join(result_path, 'edged_image.jpg'), edged)
        
        filtered_edges = keep_long_segments(edged, min_length=50)
        cv2.imwrite(os.path.join(result_path, 'filtered_edges.jpg'), filtered_edges)

        # 2. 윤곽선 찾기 및 필터링
        screenCnt = find_document_contour(image, filtered_edges)
        cv2.imwrite(os.path.join(result_path, 'contour_image.jpg'), cv2.drawContours(image.copy(), [screenCnt], -1, (0, 255, 0), 2))
        
        cv2.imshow("Detected Contour", cv2.drawContours(image.copy(), [screenCnt], -1, (0, 255, 0), 2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if screenCnt is not None:
            # 3. 원근 변환 (이 부분은 주석 처리되어 있습니다)
            pass  # 원근 변환 코드는 주석 처리되어 있습니다.
        else:
            print("윤곽선을 찾지 못해 원근 변환을 수행하지 않습니다.")
    
# else:
#     cv2.imwrite('./out/contour_image.jpg', cv2.drawContours(image.copy(), [screenCnt], -1, (0, 255, 0), 2))

#     # 3. 원근 변환
#     # 찾은 윤곽선(screenCnt)을 원본 이미지 비율에 맞게 복원
#     pts = screenCnt.reshape(4, 2) * ratio

#     # 꼭짓점 좌표를 정렬 (상단-왼쪽, 상단-오른쪽, 하단-오른쪽, 하단-왼쪽 순)
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]

#     (tl, tr, br, bl) = rect

#     # 너비와 높이 계산
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))

#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
    
#     # 변환 후의 목적지 좌표
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
    
#     # 변환 행렬 계산 및 적용
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

#     print("STEP 3: 원근 변환 완료")

#     # 결과 이미지 저장 및 표시
#     cv2.imwrite('./out/scanned_receipt.jpg', warped)
#     # cv2.imshow("Original", cv2.resize(orig, (int(orig.shape[1] / 4), int(orig.shape[0] / 4))))
#     # cv2.imshow("Scanned", cv2.resize(warped, (int(warped.shape[1] / 4), int(warped.shape[0] / 4))))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()