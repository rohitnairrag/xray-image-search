import numpy as np
import os

paths = np.load("image_paths.npy", allow_pickle=True)

new_paths = []

for p in paths:
    fname = os.path.basename(p).lower()

    if "chest" in fname:
        folder = "Chest"
    elif "dental" in fname:
        folder = "Dental"
    elif "fracture" in fname:
        folder = "Fracture"
    elif "spine" in fname:
        folder = "Spine"
    else:
        folder = "Chest"  # fallback

    new_paths.append(os.path.join("images", folder, fname))

new_paths = np.array(new_paths, dtype=object)
np.save("image_paths.npy", new_paths)

print("✅ image_paths.npy rebuilt for GitHub folder structure")
