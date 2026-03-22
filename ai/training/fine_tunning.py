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
    data_yaml = r"D:\Swimming-analysis\ai\training\dataset_1.yaml"
    pretrained_run = r"D:\Swimming-analysis\runs\train\yolo11m_swimmer_clean\weights\best.pt"

    # load previous best model
    model = YOLO(pretrained_run)

    # fine-tune
    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=40,                 # 30–50 مناسب غالبًا
        batch=8,
        project=r"D:\Swimming-analysis\runs\train",
        name="yolo11m_swimmer_finetune_v2",
        exist_ok=True,

        optimizer="SGD",           # fine-tuning final polish
        lr0=0.005,
        lrf=0.01,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,

        patience=15,
        freeze=0,
        close_mosaic=10,
        label_smoothing=0.0,

        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.25,
        degrees=3.0,
        translate=0.08,
        scale=0.25,
        shear=0.5,
        perspective=0.0002,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.10,

        device=0,
        workers=4,
        cache=False,
        amp=True,
        seed=42,
        deterministic=False,

        val=True,
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
    )

    print("Fine-tuning complete.")
    print("Best checkpoint:", model.trainer.best)
    print("Results saved under:", model.trainer.save_dir)
    print("Metrics object:", results)


if __name__ == "__main__":
    main()