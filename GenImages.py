import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path
import os
import csv

INPUT_DIR = Path("data/Simustruct_data")
OUTPUT_IMG_DIR = Path("data/stress_images")
OUTPUT_CSV = Path("data/max_stress.csv")

OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Prepare CSV to record max stress values
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "image_file", "max_stress"])

# Loop over numbered folders (1, 2, 3, ...)
for folder in sorted(INPUT_DIR.iterdir(), key=lambda x: int(x.name)):
    sample_id = folder.name

    try:
        coords = np.loadtxt(folder / "mesh_geometry.csv", delimiter=",")
        triangles = np.loadtxt(folder / "mesh_topology.csv", delimiter=",", dtype=int)
        stress = np.loadtxt(folder / "von_Mises_stress.csv", delimiter=",").flatten()

        # Create triangulation
        triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

        # Render stress field
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax.set_axis_off()
        ax.tripcolor(triang, stress, shading='gouraud', cmap='viridis')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        img_name = f"sample_{sample_id.zfill(4)}.png"
        plt.savefig(OUTPUT_IMG_DIR / img_name, dpi=100)
        plt.close()

        # Record max stress
        max_stress = float(np.max(stress))
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([sample_id, img_name, max_stress])

        print(f"Processed folder {sample_id} → {img_name}")

    except Exception as e:
        print(f"Skipped folder {sample_id}: {e}")
