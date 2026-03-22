from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def main() -> None:
    ensure_ultralytics()
    from ultralytics import YOLO

    # paths
    data_yaml = r"D:\Swimming-analysis\ai\training\dataset.yaml" # path to your dataset.yaml file

    # model
    model = YOLO("yolo11m.pt") # here you can change the base model, e.g. to "last.pt" or "best.pt"

    # train
    results = model.train(
        data=data_yaml,
        imgsz=640,# resize images to 640x640 for training, adjust if needed
        epochs=150,# number of training epochs, adjust based on your dataset size and convergence
        batch=8,# adjust based on your GPU memory, e.g. 16 or 32 for larger GPUs, or 4 for smaller ones
        project=r"D:\Swimming-analysis\runs\train", # path where training runs will be saved
        name="yolo11m_swimmer_clean", #here you can change the name of the training run folder
        exist_ok=True,

        optimizer="AdamW", # you can experiment with different optimizers like "SGD" or "AdamW"
        lr0=0.0015,# initial learning rate
        lrf=0.01,# final learning rate (lr0 * lrf)
        cos_lr=True,
        momentum=0.90,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.80,
        warmup_bias_lr=0.05,

        patience=30, 
        freeze=0,
        close_mosaic=15,
        label_smoothing=0.0,

        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.10,
        scale=0.35,
        shear=1.0,
        perspective=0.0003,
        fliplr=0.50,
        flipud=0.0,
        mosaic=0.50,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.15,

        device=0,# set to 0 or "cuda:0" to use GPU, or "cpu" to train on CPU (not recommended)
        workers=4,# number of dataloader workers, adjust based on your CPU cores and system capabilities
        cache=False,# set to True to cache images in RAM for faster training, but requires more memory
        amp=True,# use mixed precision training for faster performance on compatible GPUs
        seed=42,# set random seed for reproducibility, but note that some operations may still be non-deterministic
        deterministic=False,# you can set this to True for reproducible results, but it may slow down training

        val=True,# whether to run validation after each epoch
        save=True,# whether to save model checkpoints
        save_period=10,# save a checkpoint every 10 epochs
        plots=True,# whether to generate training plots (loss curves, metrics, etc.) at the end of training
        verbose=True,# whether to print detailed training progress and metrics to the console
    )

    print("Training complete.")
    print("Best checkpoint:", model.trainer.best)
    print("Results saved under:", model.trainer.save_dir)
    print("Metrics object:", results)


if __name__ == "__main__": # run the main function when this script is executed
    main()