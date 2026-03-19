<<<<<<< HEAD
import os

label_folder = "dataset/train/labels"
image_folder = "dataset/train/images"

for file in os.listdir(label_folder):
    path = os.path.join(label_folder, file)

    if os.path.getsize(path) == 0:
        os.remove(path)

        img = file.replace(".txt", ".jpg")
        img_path = os.path.join(image_folder, img)

        if os.path.exists(img_path):
=======
import os

label_folder = "dataset/train/labels"
image_folder = "dataset/train/images"

for file in os.listdir(label_folder):
    path = os.path.join(label_folder, file)

    if os.path.getsize(path) == 0:
        os.remove(path)

        img = file.replace(".txt", ".jpg")
        img_path = os.path.join(image_folder, img)

        if os.path.exists(img_path):
>>>>>>> c2eba27109d5ce886fe725621475426ae1166196
            os.remove(img_path)