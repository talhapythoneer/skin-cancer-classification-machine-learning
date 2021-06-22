import numpy as np
import cv2
import os

def hair_removal(img):
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (9, 9))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
    return dst


def main_segmentation(img):
    src_img = img
    # ---call function for removing hairs
    pre_processed_img = hair_removal(src_img)
    # cv2.imshow("hair_removed_2", pre_processed_img)
    # ---Extract Blue
    blue_channel = pre_processed_img[:, :, 0]
    # ---top-bottom hat filtering
    kernel_hat = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (3, 3))
    # Applying the Top-Hat operation
    top_hat_img = cv2.morphologyEx(blue_channel,
                                   cv2.MORPH_TOPHAT,
                                   kernel_hat)
    hat_img = top_hat_img + blue_channel
    # ---adjusting intensities
    max_int = np.amax(hat_img)
    min_int = np.amin(hat_img)
    diff = max_int - min_int
    stretchlim = (hat_img - min_int) / diff
    # ---contrast stretching
    norm_img1 = cv2.normalize(stretchlim, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # scale to uint8
    contrast_stretched = (255 * norm_img1).astype(np.uint8)
    # cv2.imshow("Contrast Stretched_6", contrast_stretched)
    # ---Otsu Thresholding
    # Gaussian filtering
    blur = cv2.GaussianBlur(contrast_stretched, (5, 5), 0)
    # segmentation
    ret3, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = 255 - otsu_thresh
    f_otsu_blurred = cv2.GaussianBlur(otsu_thresh, (5, 5), 0)
    # cv2.imshow("Pso_Thresh_8", otsu_thresh)
    return pre_processed_img, contrast_stretched, otsu_thresh

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

if __name__ == "__main__":
    mypath = r'C:\Users\Zayn\Desktop\DIP Research Work\Datasets\original\train\malignant'
    images, onlyfiles = load_images_from_folder(mypath)
    for i in range(len(images)):
        hair_removed, contrast, final = main_segmentation(images[i])
        cv2.imwrite(
            r"C:\Users\Zayn\Desktop\DIP Research Work\Datasets\hair_removed\train\malignant/" + onlyfiles[i]
            , hair_removed)
        cv2.imwrite(
            r"C:\Users\Zayn\Desktop\DIP Research Work\Datasets\contrast\train\malignant/" + onlyfiles[i]
            , contrast)
        cv2.imwrite(
            r"C:\Users\Zayn\Desktop\DIP Research Work\Datasets\final\train\malignant/" + onlyfiles[i]
            , final)

    cv2.waitKey(0)
