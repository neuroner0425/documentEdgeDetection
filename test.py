import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import src.docgedect as dc

DATASET_PATH = './dataset'

def detect_documents(dataset: str):
    input_dir = os.path.join(DATASET_PATH, dataset)
    if not os.path.isdir(input_dir):
        print(f"Not a directory: {input_dir}")
        return

    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

    if not img_files:
        print(f"No images found in {input_dir}")
        return

    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"Processing {img_path}...")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        save_steps_dir = os.path.join(input_dir, "preprocess_steps")
        os.makedirs(save_steps_dir, exist_ok=True)

        preprocessed = dc.preprocess_image(
            image,
            max_size=1000.0,
            reduce_lighting_=True,
            gray=True,
            contrast=3,
            exposure=-150,
            save_steps=save_steps_dir
        )

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        axes[0].imshow(dc.opencv2pil(preprocessed), cmap='gray')
        axes[0].set_title("Preprocessed")
        axes[0].axis('off')

        candy = cv2.Canny(preprocessed, 75, 150)
        axes[1].imshow(candy, cmap='gray')
        axes[1].set_title("Canny Edge")
        axes[1].axis('off')

        kernel = np.ones((1, 1), np.uint8)
        closing = cv2.morphologyEx(candy, cv2.MORPH_CLOSE, kernel)
        axes[2].imshow(closing, cmap='gray')
        axes[2].set_title("Closing")
        axes[2].axis('off')

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edge_img = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel2)
        axes[3].imshow(edge_img, cmap='gray')
        axes[3].set_title("Gradient Edge")
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

        # 윤곽선 검출
        dc.find_document_contour(dc.resize_image(image, 1000), edge_img)

if __name__ == "__main__":
    datasets = os.listdir(DATASET_PATH)
    print("Available datasets:", datasets)
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        detect_documents(dataset)
