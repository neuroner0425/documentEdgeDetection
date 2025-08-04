import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

from .lighting import reduce_lighting
from .contrast import adjust_contrast_exposure, adjust_black_point, adjust_highlight
from .resize import resize_image

def opencv2pil(opencv_image: np.ndarray) -> Image.Image:
    """OpenCV(BGR) 이미지를 PIL(RGB) 이미지로 변환"""
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_coverted)

def preprocess_image(
    image: np.ndarray,
    max_size: float = 1000.0,
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
        image = reduce_lighting(image)
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
