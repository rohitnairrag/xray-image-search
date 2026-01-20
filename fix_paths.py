import numpy as np
import os

old = np.load("image_paths.npy", allow_pickle=True)

new_paths = []
for p in old:
    fname = os.path.basename(p)
    new_paths.append(os.path.join("images", fname))

np.save("image_paths.npy", np.array(new_paths, dtype=object))

print("Done. Paths fixed.")

