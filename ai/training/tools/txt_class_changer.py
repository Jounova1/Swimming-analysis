import os

base_path = r"C:\Swimming-analysis\ai\training\Swimmer Detection.v1i.yolov11"

for split in ["train", "valid", "test"]:
    labels_folder = os.path.join(base_path, split, "labels")

    if not os.path.exists(labels_folder):
        print(f"❌ Folder not found: {labels_folder}")
        continue

    for file in os.listdir(labels_folder):
        if file.endswith(".txt"):
            file_path = os.path.join(labels_folder, file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    parts[0] = "0"   # unify to swimmer
                new_lines.append(" ".join(parts))

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))

print("✅ All labels converted to class 0 (Swimmer)")