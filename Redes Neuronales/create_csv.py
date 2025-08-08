with open("cats_dogs.csv", "w") as f:
    f.write("imagen,etiqueta\n")
    for i in range(1, 6247):
        f.write(f"cat ({i}).jpg,1\n")
    for i in range(1, 6220):
        f.write(f"dog ({i}).jpg,0\n")
    for i in range(0, 12500):
        f.write(f"cat.{i}.jpg,1\n")
    for i in range(0, 12500):
        f.write(f"dog.{i}.jpg,0\n")


#####..............COMPROBACION................######

import os
import pandas as pd

csv_file = "cats_dogs.csv"
root_dir = "CatsDogs"

df = pd.read_csv(csv_file)
missing_files = []

for img_name in df['imagen']:
    img_path = os.path.join(root_dir, img_name)
    if not os.path.isfile(img_path):
        missing_files.append(img_name)

print(f"Archivos listados en CSV pero NO encontrados: {len(missing_files)}")
print(missing_files[:20])  # muestra los primeros 20 que faltan

print("cats_dogs.csv generado correctamente")