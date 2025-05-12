import cv2
import numpy as np
import matplotlib.pyplot as plt


# =================
# Descriptor LAB
# =================

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_heatmap(img):
    # Convertir imagen a LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    umbral_a = 140
    umbral_b = 145
    umbral_l = 100

    _, thresh_a = cv2.threshold(a, umbral_a, 255, cv2.THRESH_BINARY_INV)
    _, thresh_b = cv2.threshold(b, umbral_b, 255, cv2.THRESH_BINARY_INV)
    _, mask_dark = cv2.threshold(l, umbral_l, 255, cv2.THRESH_BINARY_INV)

    # Combinar máscaras
    combined_mask = cv2.bitwise_and(thresh_a, thresh_b)

    # Aplicar máscara a la imagen original
    highlighted_image = cv2.bitwise_and(img, img, mask=mask_dark)

    # Crear heatmap
    heatmap = cv2.applyColorMap(highlighted_image, cv2.COLORMAP_JET)

    """ plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    plt.imshow(a, cmap='gray')
    plt.title("Canal 'a'")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(b, cmap='gray')
    plt.title("Canal 'b'")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    plt.tight_layout()
    plt.show()"""

    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    plt.imshow(l, cmap='gray')
    plt.title("Canal 'L' (Luminosidad)")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(mask_dark, cmap='gray')
    plt.title("Máscara de zonas oscuras")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return heatmap, mask_dark

img_path = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000\01_TUMOR\1D8F_CRC-Prim-HE-02_007b.tif_Row_151_Col_1.tif'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

heatmap, mask = create_heatmap(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# === Visualizar ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Máscara Binaria (Zonas relevantes)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(heatmap_rgb)
plt.title('Heatmap sobre Zonas Destacadas')
plt.axis('off')

plt.tight_layout()
plt.show()