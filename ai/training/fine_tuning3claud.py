"""
YOLO Custom Training Script
============================
Trains a YOLOv8/YOLO11 model on a custom dataset using Ultralytics.
Optimised for an NVIDIA RTX 3050 GPU with strong but balanced augmentations.

Usage:
    python train_yolo.py
"""

import subprocess
import sys


# ---------------------------------------------------------------------------
# Helper: ensure Ultralytics is available
# ---------------------------------------------------------------------------

def install_ultralytics() -> None:
    """
    Attempt to import Ultralytics; install it via pip if the package is
    missing.  This keeps the script self-contained so it can be dropped on a
    fresh environment without a separate setup step.
    """
    try:
        import ultralytics  # noqa: F401 – import check only
    except ImportError:
        print("[INFO] Ultralytics not found – installing via pip …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"]
        )
        print("[INFO] Ultralytics installed successfully.\n")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load a pretrained YOLO checkpoint and fine-tune it on a custom dataset.

    All hyper-parameters are documented inline so the script doubles as a
    reference card for YOLO training options.
    """

    # ------------------------------------------------------------------
    # 0.  Imports (after the optional install above)
    # ------------------------------------------------------------------
    install_ultralytics()

    from ultralytics import YOLO  # noqa: E402

    # ------------------------------------------------------------------
    # 1.  Model – load from a pretrained / previously trained checkpoint
    # ------------------------------------------------------------------
    # 'best.pt' is the checkpoint saved by a prior training run (or a
    # domain-specific fine-tune base).  Swap in 'yolov8n.pt' etc. if you
    # want to start from the official ImageNet pretrained weights instead.
    model = YOLO("best.pt")

    # ------------------------------------------------------------------
    # 2.  Dataset
    # ------------------------------------------------------------------
    # The YAML file must contain:
    #   path  – root directory of the dataset
    #   train – relative path to training images / labels
    #   val   – relative path to validation images / labels
    #   names – dict / list of class names
    DATA_YAML = "dataset.yaml"

    # ------------------------------------------------------------------
    # 3.  Hardware / IO settings  (tuned for RTX 3050 – 4 GB VRAM)
    # ------------------------------------------------------------------
    IMGSZ   = 640   # Input resolution; 640×640 is the YOLO sweet-spot and
                    # fits comfortably in 4 GB VRAM at batch=8.
    BATCH   = 8     # Mini-batch size.  Decrease to 4 if you see OOM errors
                    # or increase to 16 if you have spare VRAM headroom.
    WORKERS = 4     # DataLoader worker processes.  4 matches a typical
                    # quad-core CPU without saturating it.
    DEVICE  = 0     # GPU index (0 = first GPU).  Use 'cpu' to force CPU.
    AMP     = True  # Automatic Mixed Precision (FP16).  Halves VRAM usage
                    # and speeds up training on Ampere / Turing GPUs with
                    # almost no accuracy cost.

    # ------------------------------------------------------------------
    # 4.  Training schedule
    # ------------------------------------------------------------------
    EPOCHS       = 80   # Total training epochs.  80 is a solid default for
                        # most custom datasets; increase to 150–300 for
                        # small or highly imbalanced datasets.
    WARMUP_EPOCHS = 3   # Linear LR warm-up for the first N epochs.
                        # Prevents instability when the randomly initialised
                        # detection head starts updating with a high LR.
    PATIENCE     = 30   # Early-stopping patience (epochs without mAP
                        # improvement).  Set to 0 to disable early stopping.
    SAVE_PERIOD  = 10   # Persist a checkpoint every N epochs in addition to
                        # the automatic best.pt / last.pt saves.  Useful for
                        # post-hoc analysis and recovery from crashes.

    # ------------------------------------------------------------------
    # 5.  Optimiser – SGD with momentum
    # ------------------------------------------------------------------
    # SGD + cosine LR generally generalises better than Adam for detection
    # when trained for ≥ 50 epochs.
    OPTIMIZER      = "SGD"
    LR0            = 0.003    # Initial (peak) learning rate.  Lower than the
                              # default 0.01 because we are fine-tuning an
                              # existing checkpoint rather than training from
                              # scratch.
    LRF            = 0.01     # Final LR = LR0 × LRF.  With cosine decay and
                              # LRF=0.01, the LR drops to 3e-5 by epoch 80.
    MOMENTUM       = 0.937    # SGD momentum / Adam β₁.  0.937 is the
                              # Ultralytics default, well-validated across
                              # COCO experiments.
    WEIGHT_DECAY   = 0.0005   # L2 regularisation.  Keeps weights small and
                              # helps prevent over-fitting on small datasets.
    COSINE_LR      = True     # Use cosine annealing instead of linear decay.
                              # Produces a smoother loss curve and typically
                              # a slightly higher final mAP.

    # ------------------------------------------------------------------
    # 6.  Augmentations – strong but balanced
    # ------------------------------------------------------------------
    # Colour-space jitter
    HSV_H = 0.015   # Hue shift fraction.  Subtle shift keeps colours
                    # plausible (e.g. avoids turning red into green).
    HSV_S = 0.5     # Saturation jitter.  Simulates overcast / sunny lighting.
    HSV_V = 0.35    # Brightness jitter.  Handles under/over-exposed images.

    # Geometric transforms
    DEGREES     = 3.0     # Random rotation ± degrees.  Small value avoids
                          # distorting objects with strong orientation priors
                          # (e.g. vehicles viewed from above).
    TRANSLATE   = 0.08    # Fraction of image size for random XY translation.
    SCALE       = 0.35    # Scale jitter ± fraction; simulates distance
                          # variation between camera and object.
    SHEAR       = 0.5     # Shear angle in degrees; light shear improves
                          # robustness to camera tilt.
    PERSPECTIVE = 0.0003  # Perspective warp fraction.  Very small to stay
                          # physically plausible (≤ 0.001 recommended).

    # Flip
    FLIPLR = 0.5   # Horizontal flip probability.  Almost always beneficial
                   # unless objects have a fixed left/right orientation.
    FLIPUD = 0.0   # Vertical flip probability.  Disabled; flipping upside-
                   # down is rarely realistic for most detection tasks.

    # Advanced augmentations
    MOSAIC  = 0.25   # Mosaic probability (combines 4 images into one).
                     # Reduced from the default 1.0 because the model is
                     # already pretrained – aggressive mosaic can hurt
                     # fine-tuning on small datasets.
    MIXUP   = 0.05   # MixUp probability (blends two images + labels).
                     # Low value adds mild regularisation without muddying
                     # clear object boundaries.
    ERASING = 0.1    # Random erasing probability.  Encourages the model to
                     # use context rather than a single salient patch.

    # ------------------------------------------------------------------
    # 7.  Logging / output
    # ------------------------------------------------------------------
    PROJECT = "runs/train"   # Top-level directory for all output artefacts.
    NAME    = "custom_yolo"  # Sub-directory name; Ultralytics auto-increments
                             # (custom_yolo, custom_yolo2, …) to avoid clashes.
    PLOTS   = True           # Save training-curve PNGs (loss, mAP, PR curve).
    VAL     = True           # Run validation after every epoch and save the
                             # checkpoint that achieves the best val mAP50-95.

    # ------------------------------------------------------------------
    # 8.  Kick off training
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Starting YOLO training")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH}  |  imgsz : {IMGSZ}")
    print(f"  Device  : {DEVICE}  |  AMP   : {AMP}")
    print("=" * 60, "\n")

    results = model.train(
        # ── Data ──────────────────────────────────────────────────────
        data    = DATA_YAML,
        imgsz   = IMGSZ,

        # ── Schedule ──────────────────────────────────────────────────
        epochs          = EPOCHS,
        warmup_epochs   = WARMUP_EPOCHS,
        patience        = PATIENCE,
        save_period     = SAVE_PERIOD,

        # ── Optimiser ─────────────────────────────────────────────────
        optimizer       = OPTIMIZER,
        lr0             = LR0,
        lrf             = LRF,
        momentum        = MOMENTUM,
        weight_decay    = WEIGHT_DECAY,
        cos_lr          = COSINE_LR,

        # ── Hardware ──────────────────────────────────────────────────
        batch           = BATCH,
        workers         = WORKERS,
        device          = DEVICE,
        amp             = AMP,

        # ── Colour augmentations ──────────────────────────────────────
        hsv_h           = HSV_H,
        hsv_s           = HSV_S,
        hsv_v           = HSV_V,

        # ── Geometric augmentations ───────────────────────────────────
        degrees         = DEGREES,
        translate       = TRANSLATE,
        scale           = SCALE,
        shear           = SHEAR,
        perspective     = PERSPECTIVE,
        fliplr          = FLIPLR,
        flipud          = FLIPUD,

        # ── Advanced augmentations ────────────────────────────────────
        mosaic          = MOSAIC,
        mixup           = MIXUP,
        erasing         = ERASING,

        # ── Output ────────────────────────────────────────────────────
        project         = PROJECT,
        name            = NAME,
        plots           = PLOTS,
        val             = VAL,
    )

    # ------------------------------------------------------------------
    # 9.  Post-training summary
    # ------------------------------------------------------------------
    # results.save_dir is a pathlib.Path pointing to the run directory.
    save_dir  = results.save_dir
    best_ckpt = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Results directory : {save_dir}")
    print(f"  Best checkpoint   : {best_ckpt}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()