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
    data_yaml = r"D:\Swimming-analysis\ai\training\dataset.yaml"

    # model
    model = YOLO("yolo11m.pt")

    # train
    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=150,
        batch=8,
        project=r"D:\Swimming-analysis\runs\train",
        name="yolo11m_swimmer_clean",
        exist_ok=True,

        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
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

        device=0,
        workers=4,
        cache=False,
        amp=True,
        seed=42,
        deterministic=False,

        val=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("Training complete.")
    print("Best checkpoint:", model.trainer.best)
    print("Results saved under:", model.trainer.save_dir)
    print("Metrics object:", results)


if __name__ == "__main__":
    main()