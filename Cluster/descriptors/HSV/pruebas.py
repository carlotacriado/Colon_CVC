import cv2
import numpy as np

# Cargar imagen
img_path = r'C:\Users\ccriado\Desktop\CVC\Colon_CVC_rep\Colon_CVC\Kather_texture_2016_image_tiles_5000\03_COMPLEX\1D7_CRC-Prim-HE-01_036.tif_Row_151_Col_151.tif'
img = cv2.imread(img_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Crear ventana
cv2.namedWindow('Adjust HSV Mask')

# Función vacía para los sliders
def nothing(x):
    pass

# Crear sliders para H, S, V min y max
cv2.createTrackbar('H_min', 'Adjust HSV Mask', 0, 179, nothing)
cv2.createTrackbar('H_max', 'Adjust HSV Mask', 179, 179, nothing)
cv2.createTrackbar('S_min', 'Adjust HSV Mask', 0, 255, nothing)
cv2.createTrackbar('S_max', 'Adjust HSV Mask', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Adjust HSV Mask', 0, 255, nothing)
cv2.createTrackbar('V_max', 'Adjust HSV Mask', 255, 255, nothing)

while True:
    # Leer valores de los sliders
    h_min = cv2.getTrackbarPos('H_min', 'Adjust HSV Mask')
    h_max = cv2.getTrackbarPos('H_max', 'Adjust HSV Mask')
    s_min = cv2.getTrackbarPos('S_min', 'Adjust HSV Mask')
    s_max = cv2.getTrackbarPos('S_max', 'Adjust HSV Mask')
    v_min = cv2.getTrackbarPos('V_min', 'Adjust HSV Mask')
    v_max = cv2.getTrackbarPos('V_max', 'Adjust HSV Mask')

    # Crear máscara y aplicarla
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Mostrar resultado aplicado sobre la imagen original
    highlighted = cv2.bitwise_and(img, img, mask=mask)

    # Mostrar imagen original, máscara y resultado
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Highlighted', highlighted)

    # Pulsar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Valores finales HSV:")
        print(f"H: {h_min}-{h_max}, S: {s_min}-{s_max}, V: {v_min}-{v_max}")
        break

cv2.destroyAllWindows()
