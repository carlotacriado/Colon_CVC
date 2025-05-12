import os
import re
from collections import defaultdict

# Ruta base al dataset
base_path = r'/home/ccriado/CVC/Colon_CVC/Cluster/Kather_texture_2016_image_tiles_5000'
class_names = sorted(os.listdir(base_path))

# Extraer ID de imagen madre
def extract_image_id(filename):
    match = re.search(r'(CRC-Prim-HE-[\w\-]+\.tif)', filename)
    return match.group(1) if match else None

# Analizar por clase
for class_name in class_names:
    class_dir = os.path.join(base_path, class_name)
    image_counts = defaultdict(int)

    for fname in os.listdir(class_dir):
        if fname.endswith(".tif"):
            image_id = extract_image_id(fname)
            if image_id:
                image_counts[image_id] += 1
            else:
                print(f"Error: No se pudo extraer el ID de la imagen de {fname}")


    print(f"\n� Clase: {class_name}")
    print(f"   Total patches: {sum(image_counts.values())}")
    print(f"   Imágenes madre únicas: {len(image_counts)}")
    print(f"   Media de patches por imagen madre: {sum(image_counts.values()) / len(image_counts):.2f}\n")
    
    for img_id, count in sorted(image_counts.items(), key=lambda x: -x[1]):
        print(f"   {img_id}: {count} patches")
        