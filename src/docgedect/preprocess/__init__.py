import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def OpenCV2PIL(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image

def reduce_lighting(img):
    ### homomorphic filter는 gray scale image에 대해서 밖에 안 되므로
    ### YUV color space로 converting한 뒤 Y에 대해 연산을 진행
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    
    y = img_YUV[:,:,0]
    
    rows = y.shape[0]    
    cols = y.shape[1]
    
    ### illumination elements와 reflectance elements를 분리하기 위해 log를 취함
    imgLog = np.log1p(np.array(y, dtype='float') / 255) # y값을 0~1사이로 조정한 뒤 log(x+1)
    
    ### frequency를 이미지로 나타내면 4분면에 대칭적으로 나타나므로 
    ### 4분면 중 하나에 이미지를 대응시키기 위해 row와 column을 2배씩 늘려줌
    M = 2*rows + 1
    N = 2*cols + 1
    
    ### gaussian mask 생성 sigma = 10
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬
    Xc = np.ceil(N/2) # 올림 연산
    Yc = np.ceil(M/2)
    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성
    
    ### low pass filter와 high pass filter 생성
    LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
    HPF = 1 - LPF
    
    ### LPF랑 HPF를 0이 가운데로 오도록iFFT함. 
    ### 사실 이 부분이 잘 이해가 안 가는데 plt로 이미지를 띄워보니 shuffling을 수행한 효과가 났음
    ### 에너지를 각 귀퉁이로 모아 줌
    LPF_shift = np.fft.ifftshift(LPF.copy())
    HPF_shift = np.fft.ifftshift(HPF.copy())
    
    ### Log를 씌운 이미지를 FFT해서 LPF와 HPF를 곱해 LF성분과 HF성분을 나눔
    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분

    ### 각 LF, HF 성분에 scaling factor를 곱해주어 조명값과 반사값을 조절함
    gamma1 = 0.3
    gamma2 = 1.5
    img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]
    
    ### 조정된 데이터를 이제 exp 연산을 통해 이미지로 만들어줌
    img_exp = np.expm1(img_adjusting) # exp(x) + 1
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화
    img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌
    
    ### 마지막으로 YUV에서 Y space를 filtering된 이미지로 교체해주고 RGB space로 converting
    img_YUV[:,:,0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return result

def resize_image(image, max_size=1000.0):
    """max_size에 맞게 이미지 줄이기"""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        ratio = max_size / max(height, width)
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    return image

def adjust_contrast_exposure(image, contrast=1.0, exposure=0.0):
    """이미지에 대비와 노출 적용"""
    exposure_adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=exposure)
    return exposure_adjusted

def adjust_black_point(image, threshold=100):
    """이미지에 블랙포인트 적용"""
    black_point_adjusted = image.copy()
    black_point_adjusted[black_point_adjusted <= threshold] = 0
    return black_point_adjusted

def adjust_highlight(image, increase=100):
    """이미지에 하이라이트 효과를 적용"""
    highlight_mask = image > 200
    highlight_adjusted = image.copy()
    highlight_adjusted[highlight_mask] = np.clip(highlight_adjusted[highlight_mask] + increase, 0, 255)
    return highlight_adjusted

def preprocess_image(image,
                    max_size = 1000.0,
                    reduceLighting = False,
                    gray = False,
                    contrast=1.0, 
                    exposure=0.0, 
                    black_point_threshold=None, 
                    highlight_increase=None, 
                    show_steps=False, save_steps=None):
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    
    image = resize_image(image, max_size)
    
    if(reduceLighting):
        image = reduce_lighting(image)

    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if save_steps:
            cv2.imwrite(os.path.join(save_steps, '0.gray.jpg'), image)
    
    axes[0].imshow(OpenCV2PIL(image))
    axes[0].set_title(f'Sample {0+1}')
    axes[0].axis('off')

    image = adjust_contrast_exposure(image, contrast=contrast, exposure=exposure)
    if save_steps:
        cv2.imwrite(os.path.join(save_steps, '1.exposure_adjusted_image.jpg'), image)
        
    axes[1].imshow(OpenCV2PIL(image))
    axes[1].set_title(f'Sample {1+1}')
    axes[1].axis('off')

    if black_point_threshold is not None:
        image = adjust_black_point(image, threshold=black_point_threshold)
        if save_steps:
            cv2.imwrite(os.path.join(save_steps, '2.black_point_adjusted_image.jpg'), image)
            
    axes[2].imshow(OpenCV2PIL(image))
    axes[2].set_title(f'Sample {2+1}')
    axes[2].axis('off')

    if highlight_increase is not None:
        image = adjust_highlight(image, increase=highlight_increase)
        if save_steps:
            cv2.imwrite(os.path.join(save_steps, '3.highlight_adjusted_image.jpg'), image)
            
    axes[3].imshow(OpenCV2PIL(image))
    axes[3].set_title(f'Sample {3+1}')
    axes[3].axis('off')
    
    def on_key(event):
        plt.close()
    
    if show_steps:
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()
    else:
        plt.close()
    
    return image