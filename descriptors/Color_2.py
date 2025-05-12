import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray

image_path = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000\01_TUMOR\3EF8_CRC-Prim-HE-08_010.tif_Row_301_Col_1.tif'
image_path2 = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000\04_LYMPHO\1C49_CRC-Prim-HE-06.tif_Row_451_Col_301.tif'

image = cv2.imread(image_path)
image2 = cv2.imread(image_path2)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("rgb",image_rgb)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)


#blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
# blurred = cv2.medianBlur(gray, 5)
#blurred = cv2.bilateralFilter(gray, 9, 75, 75)


# cv2.imshow("Blur", blurred)
# cv2.waitKey(0)

# edges = cv2.Canny(gray, 100, 200)
# cv2.imshow("Canny edges", edges)
# cv2.waitKey(0)

# radius = 1
# n_points = 8 * radius
# METHOD = 'uniform'
# lbp = local_binary_pattern(gray, n_points, radius, METHOD)
# lbp_norm = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))

# # Mostrar con OpenCV
# cv2.imshow("LBP Normalizado", lbp_norm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def get_lbp_histogram(gray_img, radius=1, method='uniform'):
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2),
                           density=True)
    return hist

hist_tumor = get_lbp_histogram(gray)
hist_adipose = get_lbp_histogram(gray2)
from scipy.spatial import distance

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

dist = chi2_distance(hist_tumor, hist_adipose)
print("Distancia Chi-cuadrado:", dist)
