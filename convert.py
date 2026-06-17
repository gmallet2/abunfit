import json
from pathlib import Path

data = {}

folder = Path("data/models/SNcc/Su16/N20")

for txt_file in folder.glob("*.txt"):
    with open(txt_file, "r") as f:
        for i, line in enumerate(f, start=1):
            key = f"{txt_file.stem}_{i}"
            data[key] = [float(x) for x in line.split()]

with open("data/models/SNcc/Su16_N20.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))
