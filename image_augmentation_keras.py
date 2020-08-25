import numpy as np
import random
import cv2


def Image_Augmentation(method):
    def contrast_stretch(input_img):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        minmax_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype='uint8')

        # Loop over the image and apply Min-Max formulae
        for i in range(input_img.shape[0]):
            for j in range(input_img.shape[1]):
                minmax_img[i, j] = 255 * (input_img[i, j] - np.min(input_img)) / (np.max(input_img) - np.min(input_img))
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
        return minmax_img

    def CLAHE(input_img):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(input_img)
        cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        return cl1

    def histogram_equalization(input_img):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(input_img)
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
        return equ

    if method == "contrast_stretch":
        return contrast_stretch
    if method == "CLAHE":
        return CLAHE
    if method == "histogram_equalization":
        return histogram_equalization
