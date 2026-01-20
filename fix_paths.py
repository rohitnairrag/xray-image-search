import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

paths = []

for root, dirs, files in os.walk(IMAGES_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, BASE_DIR).replace("\\", "/")
            paths.append(rel_path)

paths = np.array(paths)

np.save("image_paths.npy", paths)

print("Saved", len(paths), "paths")
print(paths[:5])

