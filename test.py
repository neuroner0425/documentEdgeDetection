import src.docgedect as dc
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

DATASET_PATH = './dataset'
datasets = os.listdir(DATASET_PATH)

print("Available datasets:", datasets)

def detect_documents(dataset):
    img_paths = [os.path.join(DATASET_PATH, dataset, f) for f in os.listdir(os.path.join(DATASET_PATH, dataset))
                 if os.path.isfile(os.path.join(DATASET_PATH, dataset, f)) and
                 f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

    for img_path in img_paths:
        print(f"Processing {img_path}...")
        
        image = cv2.imread(img_path)
        preprocessed_image = dc.preprocess_image(image,
                                                        max_size=1000.0,
                                                        reduceLighting = True,
                                                        gray=True,
                                                        contrast=3,
                                                        exposure=-150,
                                                        black_point_threshold=None,
                                                        highlight_increase=None,
                                                        show_steps=False, 
                                                        save_steps=f'{DATASET_PATH}/{dataset}/preprocess.jpg')
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 5))
        
        axes[0].imshow(dc.OpenCV2PIL(preprocessed_image))
        axes[0].set_title(f'Sample {0+1}')
        axes[0].axis('off')
        
        candy_image = cv2.Canny(preprocessed_image, 75, 150)
        
        axes[1].imshow(dc.OpenCV2PIL(candy_image))
        axes[1].set_title(f'Sample {1+1}')
        axes[1].axis('off')
        
        kernel = np.ones((1,1
                          ), np.uint8)
        closing_image = cv2.morphologyEx(candy_image, cv2.MORPH_CLOSE, kernel)
        
        axes[2].imshow(dc.OpenCV2PIL(closing_image))
        axes[2].set_title(f'Sample {2+1}')
        axes[2].axis('off')
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edge_img = cv2.morphologyEx(closing_image, cv2.MORPH_GRADIENT, kernel)
        
        axes[3].imshow(dc.OpenCV2PIL(edge_img))
        axes[3].set_title(f'Sample {3+1}')
        axes[3].axis('off')
        
        def on_key(event):
            if event.key == 'Q' or event.key == 'q':
                sys.exit()
            else:
                plt.close()
            
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        manager = plt.get_current_fig_manager()
        try:
            manager.full_screen_toggle()
        except AttributeError:
            try:
                manager.window.state('zoomed')
            except Exception:
                pass

        plt.show()
        
        dc.find_document_contour(dc.resize_image(image, 1000), edge_img)
        
if __name__ == "__main__":
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        detect_documents(dataset)