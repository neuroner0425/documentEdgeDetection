from .preprocess import *

def find_document_contour(original_image, edged):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 찾은 윤곽선들을 크기 순으로 정렬
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    
    print(f"STEP 2: 윤곽선 찾기 시작, 총 {len(contours)}개의 윤곽선이 발견되었습니다.")
    
    copy_original_image= original_image.copy()
    for c in contours:
    
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

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