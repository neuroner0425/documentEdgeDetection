import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

from .lighting import reduce_lighting
from .contrast import adjust_contrast_exposure, adjust_black_point, adjust_highlight
from .resize import resize_image, add_padding

def opencv2pil(opencv_image: np.ndarray) -> Image.Image:
    """OpenCV(BGR) 이미지를 PIL(RGB) 이미지로 변환"""
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_coverted)

def preprocess_image(
    image: np.ndarray,
    max_size: float = 1000.0,
    padding: int = 0,
    reduce_lighting_: bool = False,
    gray: bool = False,
    contrast: float = 1.0,
    exposure: float = 0.0,
    black_point_threshold: int = None,
    highlight_increase: int = None,
    show_steps: bool = False,
    save_steps: str = None
) -> np.ndarray:
    """
    일련의 이미지 전처리 수행
    """
    steps = []
    image = resize_image(image, max_size)
    steps.append(('Resized', image.copy()))

    if reduce_lighting_:
        image = add_padding(image=image, top=padding, bottom=padding, left=padding, right=padding)
        image = reduce_lighting(image)
        h, w = image.shape[:2]
        image = image[padding:h-padding, padding:w-padding] if padding > 0 else image
        steps.append(('Lighting Reduced', image.copy()))

    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        steps.append(('Gray', image.copy()))
        if save_steps:
            cv2.imwrite(os.path.join(save_steps, 'gray.jpg'), image)

    image = adjust_contrast_exposure(image, contrast=contrast, exposure=exposure)
    steps.append(('Contrast/Exposure', image.copy()))
    if save_steps:
        cv2.imwrite(os.path.join(save_steps, 'contrast_exposure.jpg'), image)

    if black_point_threshold is not None:
        image = adjust_black_point(image, threshold=black_point_threshold)
        steps.append(('Black Point', image.copy()))
        if save_steps:
            cv2.imwrite(os.path.join(save_steps, 'black_point.jpg'), image)

    if highlight_increase is not None:
        image = adjust_highlight(image, increase=highlight_increase)
        steps.append(('Highlight', image.copy()))
        if save_steps:
            cv2.imwrite(os.path.join(save_steps, 'highlight.jpg'), image)

    if show_steps:
        fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
        if len(steps) == 1:
            axes = [axes]
        for ax, (title, step_img) in zip(axes, steps):
            if len(step_img.shape) == 2:  # gray
                ax.imshow(step_img, cmap='gray')
            else:
                ax.imshow(opencv2pil(step_img))
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return image

def realtime_preprocess_image(
    image: np.ndarray,
    initial_max_size: float = 1000.0,
    initial_padding: int = 0,
):
    """
    Matplotlib 슬라이더/체크박스로 실시간 전처리 파라미터 조절 인터랙티브 UI를 실행합니다.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, CheckButtons

    # 원본 이미지 copy (슬라이더 값에 따라 원본에서 항상 재전처리)
    orig_img = image.copy()

    # matplotlib UI 레이아웃 (아래쪽에 위젯 배치 공간 확보)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.38)

    # 초기 전처리 결과
    proc_img = preprocess_image(
        orig_img,
        max_size=initial_max_size,
        padding=initial_padding,
        reduce_lighting_=False,
        gray=False,
        contrast=1.0,
        exposure=0.0,
        black_point_threshold=None,
        highlight_increase=None,
        show_steps=False,
        save_steps=None
    )
    imshow_kwargs = {'cmap': 'gray'} if len(proc_img.shape) == 2 else {}
    im = ax.imshow(
        proc_img if len(proc_img.shape) == 2 else cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB),
        **imshow_kwargs
    )
    ax.set_title('실시간 전처리 결과')
    ax.axis('off')

    # --- 위젯들 생성 ---
    ax_contrast = plt.axes([0.35, 0.28, 0.55, 0.03])
    ax_exposure = plt.axes([0.35, 0.23, 0.55, 0.03])
    ax_padding = plt.axes([0.35, 0.18, 0.55, 0.03])
    ax_checkbox = plt.axes([0.025, 0.6, 0.2, 0.28])

    slider_contrast = Slider(ax_contrast, 'Contrast', 0.5, 3.0, valinit=1.0)
    slider_exposure = Slider(ax_exposure, 'Exposure', -150, 150, valinit=0.0)
    slider_padding = Slider(ax_padding, 'Padding', 0, 100, valinit=initial_padding, valstep=1)

    checkbox = CheckButtons(
        ax_checkbox,
        ['Reduce Lighting', 'Gray', 'Black Point', 'Highlight'],
        [False, False, False, False]
    )

    # --- 입력값 저장용 ---
    state = {
        'contrast': 1.0,
        'exposure': 0.0,
        'padding': initial_padding,
        'reduce_lighting': False,
        'gray': False,
        'black_point': False,
        'highlight': False,
    }

    # --- 슬라이더/체크박스 변경 이벤트 핸들러 ---
    def update(val=None):
        state['contrast'] = slider_contrast.val
        state['exposure'] = slider_exposure.val
        state['padding'] = int(slider_padding.val)
        cbs = checkbox.get_status()
        state['reduce_lighting'] = cbs[0]
        state['gray'] = cbs[1]
        state['black_point'] = cbs[2]
        state['highlight'] = cbs[3]

        # 블랙포인트/하이라이트는 예시로 임의 값 적용
        black_point_threshold = 100 if state['black_point'] else None
        highlight_increase = 100 if state['highlight'] else None

        out = preprocess_image(
            orig_img,
            max_size=initial_max_size,
            padding=state['padding'],
            reduce_lighting_=state['reduce_lighting'],
            gray=state['gray'],
            contrast=state['contrast'],
            exposure=state['exposure'],
            black_point_threshold=black_point_threshold,
            highlight_increase=highlight_increase,
            show_steps=False,
            save_steps=None
        )
        # 컬러/흑백 모두 처리
        if len(out.shape) == 2:
            im.set_data(out)
            im.set_cmap('gray')
        else:
            im.set_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            im.set_cmap(None)
        fig.canvas.draw_idle()

    # 슬라이더, 체크박스에 이벤트 연결
    slider_contrast.on_changed(update)
    slider_exposure.on_changed(update)
    slider_padding.on_changed(update)
    checkbox.on_clicked(update)

    plt.show()
