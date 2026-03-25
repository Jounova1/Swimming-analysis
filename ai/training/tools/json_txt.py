import os
import json

base_path = r"C:\Swimming-analysis\ai\training\annotations"
output_base = r"C:\Swimming-analysis\ai\training\data\labels"

sets = {
    "Trainset": "train",
    "Valset": "val",
    "Testset": "test"
}

class_map = {
    "swimmer": 0,
    "cap": 0
}

for json_folder, out_folder in sets.items():
    input_folder = os.path.join(base_path, json_folder)
    output_folder = os.path.join(output_base, out_folder)

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(input_folder, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        txt_lines = []

        for ann in data.get("annotations", []):

            # ✅ التصحيح هنا
            if ann.get("geometry", {}).get("type") != "rectangle":
                continue

            vertices = ann["geometry"]["vertices"]
            x1, y1, x2, y2 = vertices

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            label = ann.get("category", "").lower()
            cls = class_map.get(label, 0)

            txt_lines.append(f"{cls} {x_center} {y_center} {width} {height}")

        txt_name = file.replace(".json", ".txt")
        txt_path = os.path.join(output_folder, txt_name)

        with open(txt_path, "w") as f:
            f.write("\n".join(txt_lines))

print("✅ JSON → YOLO conversion done")