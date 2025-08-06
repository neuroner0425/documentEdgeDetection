from .preprocess import (
    preprocess_image,
    realtime_preprocess_image, opencv2pil
)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

def realtime_edge_dect(image):
    orig_img = image.copy()
    resize_img = preprocess.resize_image(orig_img)

    # 큰 그림(한 창)에 왼쪽에 이미지, 오른쪽에 컨트롤 패널
    fig = plt.figure(figsize=(12, 7))
    # 이미지 영역: 왼쪽 60%
    ax_img1 = fig.add_axes([0.05, 0.55, 0.30, 0.4])
    ax_img2 = fig.add_axes([0.40, 0.55, 0.30, 0.4])
    ax_img3 = fig.add_axes([0.05, 0.05, 0.30, 0.4])
    ax_img4 = fig.add_axes([0.40, 0.05, 0.30, 0.4])
    # 컨트롤 영역: 오른쪽 35%
    ax_contrast = fig.add_axes([0.75, 0.88, 0.20, 0.04])
    ax_exposure = fig.add_axes([0.75, 0.83, 0.20, 0.04])
    ax_padding = fig.add_axes([0.75, 0.78, 0.20, 0.04])
    ax_canny1 = fig.add_axes([0.75, 0.73, 0.20, 0.04])
    ax_canny2 = fig.add_axes([0.75, 0.68, 0.20, 0.04])
    ax_closing = fig.add_axes([0.75, 0.63, 0.20, 0.04])
    ax_gradient = fig.add_axes([0.75, 0.58, 0.20, 0.04])
    ax_checkbox = fig.add_axes([0.75, 0.20, 0.20, 0.35])

    # 슬라이더/체크박스 초기값
    slider_contrast = Slider(ax_contrast, 'Contrast', 0.5, 3.0, valinit=1.0)
    slider_contrast.set_val(val=1.8)
    slider_exposure = Slider(ax_exposure, 'Exposure', -255, 255, valinit=0.0)
    slider_exposure.set_val(val=-100)
    slider_padding = Slider(ax_padding, 'Padding', 0, 100, valinit=0, valstep=1)
    slider_padding.set_val(val=100)
    slider_canny1 = Slider(ax_canny1, 'Canny th1', 0, 600, valinit=150, valstep=1)
    slider_canny2 = Slider(ax_canny2, 'Canny th2', 0, 600, valinit=200, valstep=1)
    slider_closing = Slider(ax_closing, 'Closing k', 1, 20, valinit=1, valstep=1)
    slider_closing.set_val(val=5)
    slider_gradient = Slider(ax_gradient, 'Grad k', 1, 10, valinit=2, valstep=1)

    checkbox_labels = [
        'Reduce Lighting', 'Gray', 'Black Point', 'Highlight',
        'Use Closing'
    ]
    checkbox = CheckButtons(
        ax_checkbox,
        checkbox_labels,
        [True, False, False, False, True]
    )

    # 첫 결과 계산 및 이미지 표시
    def _preproc():
        return preprocess_image(
            orig_img,
            max_size=1000.0,
            padding=slider_padding.val,
            reduce_lighting_=True,
            gray=False,
            contrast=slider_contrast.val,
            exposure=slider_exposure.val,
            black_point_threshold=None,
            highlight_increase=None,
            show_steps=False,
            save_steps=None
        )
    proc_img = _preproc()
    ax_img1.set_title("Original")
    ax_img2.set_title("Preprocess")
    ax_img3.set_title("Edge")
    ax_img4.set_title("Countour")
    ax_img1.axis('off')
    ax_img2.axis('off')
    ax_img3.axis('off')
    ax_img4.axis('off')

    im_origin = ax_img1.imshow(opencv2pil(resize_img))
    im_proc = ax_img2.imshow(
        proc_img if len(proc_img.shape) == 2 else cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB),
        cmap='gray' if len(proc_img.shape) == 2 else None
    )
    gray_for_canny = proc_img if len(proc_img.shape) == 2 else cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_for_canny, slider_canny1.val, slider_canny2.val)
    
    k = np.ones((slider_closing.val, slider_closing.val), np.uint8)
    close_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, k)
    
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (slider_gradient.val, slider_gradient.val))
    grad_img = cv2.morphologyEx(close_img, cv2.MORPH_GRADIENT, k2)
    
    im_canny = ax_img3.imshow(grad_img, cmap='gray')
    
    cnt, best_contour, contours = find_document_contour(resize_img, grad_img)
    draw_all_contour_img = cv2.drawContours(resize_img.copy(), contours, -1, (0, 255, 0), 2)
    draw_best_contour_img = cv2.drawContours(draw_all_contour_img, [cnt], -1, (0, 0, 255), 2)
    draw_best_contour_img = cv2.drawContours(draw_best_contour_img, [best_contour], -1, (255, 0, 0), 2)
    im_countour = ax_img4.imshow(opencv2pil(draw_best_contour_img))

    def update(val=None):
        vals = {
            'contrast': slider_contrast.val,
            'exposure': slider_exposure.val,
            'padding': int(slider_padding.val),
            'canny1': int(slider_canny1.val),
            'canny2': int(slider_canny2.val),
            'closing': int(slider_closing.val),
            'gradient': int(slider_gradient.val),
        }
        cbs = checkbox.get_status()
        vals['reduce_lighting'] = cbs[0]
        vals['gray'] = cbs[1]
        vals['black_point'] = cbs[2]
        vals['highlight'] = cbs[3]
        vals['use_closing'] = cbs[4]

        out = preprocess_image(
            orig_img,
            max_size=1000.0,
            padding=vals['padding'],
            reduce_lighting_=vals['reduce_lighting'],
            gray=vals['gray'],
            contrast=vals['contrast'],
            exposure=vals['exposure'],
            black_point_threshold=100 if vals['black_point'] else None,
            highlight_increase=100 if vals['highlight'] else None,
            show_steps=False,
            save_steps=None
        )
        if len(out.shape) == 2:
            im_proc.set_data(out)
            im_proc.set_cmap('gray')
            gray_for_canny = out
        else:
            im_proc.set_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            im_proc.set_cmap(None)
            gray_for_canny = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        canny_img = cv2.Canny(gray_for_canny, vals['canny1'], vals['canny2'])
        if vals['use_closing']:
            k = np.ones((vals['closing'], vals['closing']), np.uint8)
            close_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, k)
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (vals['gradient'], vals['gradient']))
            grad_img = cv2.morphologyEx(close_img, cv2.MORPH_GRADIENT, k2)
        else:
            grad_img = close_img = canny_img

        im_canny.set_data(grad_img)
        im_canny.set_cmap('gray')
        
        cnt, best_contour, contours = find_document_contour(resize_img, grad_img)
        draw_all_contour_img = cv2.drawContours(resize_img.copy(), contours, -1, (0, 255, 0), 2)
        draw_best_contour_img = cv2.drawContours(draw_all_contour_img, [cnt], -1, (0, 0, 255), 2)
        draw_best_contour_img = cv2.drawContours(draw_best_contour_img, [best_contour], -1, (255, 0, 0), 2)
        im_countour.set_data(opencv2pil(draw_best_contour_img))
        
        fig.canvas.draw_idle()

    # 이벤트 연결
    slider_contrast.on_changed(update)
    slider_exposure.on_changed(update)
    slider_padding.on_changed(update)
    slider_canny1.on_changed(update)
    slider_canny2.on_changed(update)
    slider_closing.on_changed(update)
    slider_gradient.on_changed(update)
    checkbox.on_clicked(update)

    plt.show()


def find_document_contour(original_image: np.ndarray, edged: np.ndarray, save_dir: str = None, show_all: bool = False):
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

    # print(f"STEP 2: 윤곽선 찾기 시작, 총 {len(contours)}개 발견.")

    screen_cnt = None
    best_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            continue
        
        solidity = float(area) / hull_area

        if show_all:
            img_show = original_image.copy()
            color = (0, 255, 0)
            thickness = 2
            
            cv2.drawContours(img_show, [c], -1, color, thickness)

            h, w = img_show.shape[:2]
            text = f"solidity: {solidity:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = (w - text_width) // 2
            text_y = h - 10  # 아래에서 10픽셀 위에 표시

            cv2.putText(img_show, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)

            cv2.imshow('Contour with Solidity', img_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and solidity >= 0.9:
            screen_cnt = approx
            best_contour = c
            break

    if screen_cnt is None:
        # print("문서의 윤곽선을 찾지 못했습니다.")
        return None, None, contours
    else:
        # print("STEP 2: 윤곽선 찾기 완료")
        if(save_dir):
            os.makedirs(os.path.join(os.pardir, save_dir), exist_ok=True)
            draw_cnt = cv2.drawContours(original_image.copy(), [screen_cnt], -1, (0, 255, 0), 2)
            draw_contour = cv2.drawContours(draw_cnt, [best_contour], -1, (255, 0, 0), 2)
            cv2.imwrite(save_dir, draw_contour)
            
        return screen_cnt, best_contour, contours
