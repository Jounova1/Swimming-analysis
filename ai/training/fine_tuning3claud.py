"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           YOLO Swimmer Detection – Custom Training Script                   ║
║                                                                              ║
║  Task   : Single-class object detection  →  "Swimmer"                       ║
║  Camera : Side-view, tracking swimmers across a pool                         ║
║  Data   : Video frames (repeated), varying brightness, motion blur           ║
║  Goal   : Generalise to unseen swimmers/lighting – NOT memorise training     ║
║           frames.                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
    python train_swimmer_yolo.py

Switch between fine-tuning and training from scratch by toggling the
PRETRAINED flag inside main().
"""

import subprocess
import sys


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Dependency helper
# ─────────────────────────────────────────────────────────────────────────────

def ensure_ultralytics() -> None:
    """
    Import Ultralytics silently; install it via pip if it is not present.

    Keeping this self-contained means the script runs on a fresh environment
    without a separate 'pip install' step – convenient for lab machines or
    cloud VMs spun up just for training.
    """
    try:
        import ultralytics  # noqa: F401
        print(f"[✓] Ultralytics {ultralytics.__version__} is ready.")
    except ImportError:
        print("[!] Ultralytics not found – installing via pip …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"]
        )
        print("[✓] Ultralytics installed successfully.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Configure and launch YOLO training for swimmer detection.

    Every hyper-parameter is explained inline so this script also serves as
    a decision log – future team members can understand *why* each value was
    chosen, not just *what* it is.
    """

    ensure_ultralytics()
    from ultralytics import YOLO

    # ──────────────────────────────────────────────────────────────────────
    # 1-A.  Checkpoint selection
    # ──────────────────────────────────────────────────────────────────────
    # PRETRAINED = True  →  fine-tune from your own best.pt
    #                        (faster convergence, higher starting mAP,
    #                         ideal when you already have a decent model)
    #
    # PRETRAINED = False →  start from the official YOLO11-Medium weights
    #                        (better if best.pt was trained on very different
    #                         data or if you suspect it has overfit badly)

    PRETRAINED = True

    if PRETRAINED:
        CHECKPOINT = "best.pt"          # Your previously trained checkpoint
        print("[→] Fine-tuning from pretrained checkpoint: best.pt")
    else:
        CHECKPOINT = "yolo11m.pt"       # YOLO11-Medium, ImageNet pretrained
        print("[→] Training from official YOLO11m weights.")

    model = YOLO(CHECKPOINT)

    # ──────────────────────────────────────────────────────────────────────
    # 1-B.  Dataset
    # ──────────────────────────────────────────────────────────────────────
    # Your dataset.yaml must specify:
    #   path  : root folder of the dataset
    #   train : subfolder / txt file listing training images
    #   val   : subfolder / txt file listing validation images
    #   names : {0: 'Swimmer'}
    #
    # TIP – to fight overfitting from repeated video frames, shuffle your
    # frame list randomly *before* the train/val split so that consecutive
    # frames from the same clip are not all grouped into train or val.
    DATA_YAML = "swimmer_dataset.yaml"

    # ──────────────────────────────────────────────────────────────────────
    # 1-C.  Hardware & I/O  (tuned for RTX 3050 – 4 GB VRAM)
    # ──────────────────────────────────────────────────────────────────────
    IMGSZ = 640
    # 640 × 640 is the YOLO sweet spot: large enough to resolve partially
    # submerged swimmers and pool-lane markers without exceeding VRAM.

    BATCH = 8
    # Safe upper limit for 4 GB VRAM at imgsz=640 with AMP enabled.
    # Drop to 4 if you see CUDA out-of-memory; raise to 16 if you have a
    # larger GPU.

    WORKERS = 4
    # DataLoader background processes.  Matches a typical quad-core CPU;
    # increase if your CPU has more cores and disk I/O is the bottleneck.

    DEVICE = 0
    # 0 = first CUDA GPU.  Set to "cpu" to force CPU (slow but useful for
    # debugging).  Multi-GPU: "0,1" or list [0, 1].

    AMP = True
    # Automatic Mixed Precision (FP16 compute, FP32 master weights).
    # On RTX 3050 this cuts VRAM usage ~40 % and speeds up training ~30 %
    # with negligible accuracy loss thanks to Ampere's tensor cores.

    # ──────────────────────────────────────────────────────────────────────
    # 1-D.  Training schedule
    # ──────────────────────────────────────────────────────────────────────
    EPOCHS = 80
    # 80 epochs is sufficient for a single-class detector starting from a
    # pretrained checkpoint.  The cosine LR schedule squeezes the last
    # accuracy out of the later epochs.

    WARMUP_EPOCHS = 3
    # Ramp LR linearly from ~0 → lr0 for the first 3 epochs.
    # Critical when fine-tuning: the randomly-re-initialised detection head
    # would otherwise send large gradients back into the pretrained backbone
    # during epoch 1, potentially destroying learned features.

    PATIENCE = 25
    # Stop training early if val mAP50-95 does not improve for 25 consecutive
    # epochs.  With 80 epochs total this gives the model time to escape
    # plateaus while still preventing wasted GPU time on a diverged run.
    # Set to 0 to disable early stopping entirely.

    SAVE_PERIOD = 10
    # Write a numbered checkpoint (epoch10.pt, epoch20.pt …) in addition to
    # the automatic best.pt and last.pt.  Protects against crashes and lets
    # you roll back to an earlier epoch if the final model overfit.

    CLOSE_MOSAIC = 15
    # Disable mosaic augmentation for the LAST N epochs.
    # In the final stretch, mosaic's aggressive cropping can confuse the
    # model just as it is converging on clean, full-frame detections.
    # Switching to standard augmentation in the last 15 epochs consistently
    # lifts final mAP by 1-3 %.

    # ──────────────────────────────────────────────────────────────────────
    # 1-E.  Optimiser – SGD with momentum
    # ──────────────────────────────────────────────────────────────────────
    # Why SGD over Adam?
    # Adam converges faster in early epochs but SGD + cosine LR typically
    # reaches a flatter, better-generalising minimum given ≥ 50 epochs.
    # For a single-class detector on a relatively small video dataset,
    # generalisation matters more than raw convergence speed.

    OPTIMIZER    = "SGD"

    LR0          = 0.003
    # Peak learning rate.  Lower than the default 0.01 because we are
    # fine-tuning – we want to *adapt* the pretrained weights, not
    # overwrite them.  If training from scratch (PRETRAINED=False) you
    # could raise this to 0.007–0.01.

    LRF          = 0.01
    # Final LR multiplier → final_lr = LR0 × LRF = 3e-5.
    # With cosine decay over 80 epochs this gives a smooth, gradual
    # cool-down rather than an abrupt drop.

    MOMENTUM     = 0.937
    # SGD momentum (also β₁ for Adam).  The Ultralytics default, validated
    # extensively across COCO benchmarks.

    WEIGHT_DECAY = 0.0005
    # L2 regularisation.  Keeps weight magnitudes small, which is the main
    # regulariser on top of augmentation for a model trained on repeated
    # video frames.

    COS_LR = True
    # Cosine annealing LR schedule.  Decays LR along a cosine curve instead
    # of linearly, producing a slow start, fast middle, and slow end.
    # This shape aligns well with how loss landscapes are typically shaped.

    # ──────────────────────────────────────────────────────────────────────
    # 1-F.  Augmentation strategy
    # ──────────────────────────────────────────────────────────────────────
    # The dataset has THREE specific challenges that drive these choices:
    #   A) Repeated video frames → model can memorise exact frames → overfit
    #   B) Brightness variations → poor generalisation to different sessions
    #   C) Camera motion + perspective shifts → rigid detectors fail on edges

    # ── Colour / brightness (addresses challenge B) ──────────────────────
    HSV_H = 0.015
    # Hue jitter (± 1.5 % of the full hue wheel).
    # Subtle shift: pool water and lane markings have consistent hues;
    # larger shifts could produce unnatural colours.

    HSV_S = 0.5
    # Saturation jitter (± 50 %).
    # Simulates overcast, shaded, and artificially-lit pool environments.
    # High saturation variation is the single most important colour aug
    # for indoor/outdoor sports where lighting changes drastically.

    HSV_V = 0.35
    # Brightness (Value) jitter (± 35 %).
    # Models the full range from dim underwater lighting to bright
    # sun-lit outdoor pools.  This is the key parameter for lighting
    # robustness in swimmer detection.

    # ── Geometric transforms (addresses challenge C) ──────────────────────
    DEGREES     = 3.0
    # Rotation ± 3°.  Small: swimmers are always roughly horizontal;
    # larger rotations would create unrealistic training examples.

    TRANSLATE   = 0.08
    # XY translation up to 8 % of image size.
    # Simulates the swimmer drifting toward the edge of the camera frame
    # during fast turns, which is common in side-view tracking footage.

    SCALE       = 0.35
    # Scale jitter ± 35 %.
    # Handles swimmers at different distances from the camera (near lane
    # vs. far lane) and different zoom levels across camera rigs.

    SHEAR       = 0.5
    # Shear ± 0.5°.  Light shear improves robustness to slight camera
    # tilt that occurs when the tripod is not perfectly level.

    PERSPECTIVE = 0.0003
    # Perspective warp (fraction of image diagonal).
    # Very small value: adds subtle keystone distortion to simulate
    # cameras mounted at slightly different angles.  Values > 0.001 tend
    # to look unphysical for pool footage.

    # ── Flipping ─────────────────────────────────────────────────────────
    FLIPLR = 0.5
    # Horizontal flip with 50 % probability.
    # Doubles effective dataset size: a swimmer moving left is visually
    # equivalent to one moving right.

    FLIPUD = 0.0
    # Vertical flip disabled.
    # Upside-down swimmers do not exist in real pools; enabling this
    # would pollute the training distribution with impossible images.

    # ── Advanced augmentations (addresses challenge A: repeated frames) ───
    MOSAIC = 0.25
    # Mosaic probability: combine 4 random images into a 2×2 grid.
    # Reduced from the default 1.0 to 0.25 for two reasons:
    #   1. The model is pretrained – aggressive mosaic fights fine-tuning.
    #   2. Repeated frames make mosaic very likely to mix near-identical
    #      frames, reducing its regularisation benefit.
    # 0.25 keeps some multi-scale context learning without overpowering
    # the other augmentations.

    MIXUP = 0.05
    # MixUp probability: blend two images with a random α weight.
    # Low value (5 %) adds mild label-smoothing-like regularisation.
    # Higher values blur object boundaries, which hurts localisation
    # for partially-submerged swimmers where edges are already soft.

    ERASING = 0.1
    # Random rectangular erasing probability (10 %).
    # Acts as an occlusion simulator: in real pool footage, water splash,
    # lane ropes, and other swimmers can partially hide the target.
    # This teaches the model to fire on partial views rather than
    # requiring a fully visible swimmer silhouette.

    # ──────────────────────────────────────────────────────────────────────
    # 1-G.  Output / logging
    # ──────────────────────────────────────────────────────────────────────
    PROJECT = "runs/swimmer"
    # Parent directory for all training artefacts.

    NAME = "swimmer_detection"
    # Run sub-directory.  Ultralytics auto-increments (swimmer_detection2, …)
    # if the name already exists, so old runs are never accidentally overwritten.

    VAL   = True    # Evaluate on the validation set after every epoch and
                    # track best.pt by mAP50-95.
    PLOTS = True    # Save PNG curves: loss, mAP, precision/recall, confusion
                    # matrix.  Invaluable for diagnosing over/underfitting.

    # ──────────────────────────────────────────────────────────────────────
    # 1-H.  Launch
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  🏊  Swimmer Detection – YOLO Training")
    print("═" * 62)
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Epochs     : {EPOCHS}  |  Batch : {BATCH}  |  imgsz : {IMGSZ}")
    print(f"  Device     : cuda:{DEVICE}  |  AMP : {AMP}")
    print(f"  Optimizer  : {OPTIMIZER}  |  lr0 : {LR0}  |  cos_lr : {COS_LR}")
    print("═" * 62 + "\n")

    results = model.train(
        # ── Data ─────────────────────────────────────────────────────────
        data            = DATA_YAML,
        imgsz           = IMGSZ,

        # ── Schedule ─────────────────────────────────────────────────────
        epochs          = EPOCHS,
        warmup_epochs   = WARMUP_EPOCHS,
        patience        = PATIENCE,
        save_period     = SAVE_PERIOD,
        close_mosaic    = CLOSE_MOSAIC,

        # ── Optimiser ────────────────────────────────────────────────────
        optimizer       = OPTIMIZER,
        lr0             = LR0,
        lrf             = LRF,
        momentum        = MOMENTUM,
        weight_decay    = WEIGHT_DECAY,
        cos_lr          = COS_LR,

        # ── Hardware ─────────────────────────────────────────────────────
        batch           = BATCH,
        workers         = WORKERS,
        device          = DEVICE,
        amp             = AMP,

        # ── Colour augmentation ──────────────────────────────────────────
        hsv_h           = HSV_H,
        hsv_s           = HSV_S,
        hsv_v           = HSV_V,

        # ── Geometric augmentation ───────────────────────────────────────
        degrees         = DEGREES,
        translate       = TRANSLATE,
        scale           = SCALE,
        shear           = SHEAR,
        perspective     = PERSPECTIVE,
        fliplr          = FLIPLR,
        flipud          = FLIPUD,

        # ── Advanced augmentation ────────────────────────────────────────
        mosaic          = MOSAIC,
        mixup           = MIXUP,
        erasing         = ERASING,

        # ── Output ───────────────────────────────────────────────────────
        project         = PROJECT,
        name            = NAME,
        val             = VAL,
        plots           = PLOTS,
    )

    # ──────────────────────────────────────────────────────────────────────
    # 1-I.  Post-training summary
    # ──────────────────────────────────────────────────────────────────────
    save_dir  = results.save_dir                    # pathlib.Path
    best_ckpt = save_dir / "weights" / "best.pt"

    print("\n" + "═" * 62)
    print("  ✅  Training complete!")
    print(f"  Results directory : {save_dir}")
    print(f"  Best checkpoint   : {best_ckpt}")
    print("═" * 62 + "\n")

    print("  Next steps:")
    print("  ┌─ Validate  →  yolo val   model=best.pt data=swimmer_dataset.yaml")
    print("  ├─ Predict   →  yolo predict model=best.pt source=<video.mp4>")
    print("  └─ Export    →  yolo export model=best.pt format=onnx\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point guard
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()