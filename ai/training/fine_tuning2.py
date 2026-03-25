"""Train YOLO for swimmer detection with generalization-focused settings.

This script is configured for an RTX 3050 target profile and uses
carefully balanced augmentations for robust detection training.
"""

from __future__ import annotations  # بيسمح باستخدام type hints بشكل متقدم

import argparse        # لاستقبال arguments من الـ command line
import importlib       # للتحقق إن مكتبة معينة موجودة أو لا
import subprocess      # لتشغيل أوامر النظام زي pip install
import sys             # للوصول لـ Python interpreter الحالي
from pathlib import Path  # للتعامل مع مسارات الملفات بشكل أذكى


def install_ultralytics_if_missing() -> None:
    """Install Ultralytics if missing, or upgrade to latest if already present."""

    if importlib.util.find_spec("ultralytics") is None:
        # المكتبة مش موجودة → نثبتها من الصفر
        print("ultralytics not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("ultralytics installed successfully.")
        return

    # المكتبة موجودة → نحدثها لآخر إصدار عشان نستفيد من أحدث الـ features والـ bug fixes
    print("ultralytics found. Upgrading to latest version...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
    print("ultralytics upgraded successfully.")


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve relative paths against the script directory for convenience."""
    path = Path(path_str)
    if path.is_absolute():
        return path  # مسار كامل → مش محتاج تعديل
    return (base_dir / path).resolve()  # مسار نسبي → نكمله من مجلد السكريبت


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments so switching between training modes is easy."""

    parser = argparse.ArgumentParser(description="Ultralytics YOLO swimmer training")

    parser.add_argument(
        "--mode",
        choices=["finetune", "scratch"],
        default="finetune",
        help=(
            "finetune: resume from best.pt (faster, higher starting mAP).\n"
            "scratch: start from yolo11m.pt pretrained backbone."
        ),
    )

    parser.add_argument(
        "--data",
        default="dataset_1.yaml",
        help="Path to dataset YAML file (train/val/test paths + class names).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point: configure and launch YOLO training."""

    args = parse_args()
    install_ultralytics_if_missing()

    # نستورد YOLO بس بعد ما نتأكد إن المكتبة اتثبتت
    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    # مجلد السكريبت نفسه → هنحل منه المسارات النسبية
    script_dir = Path(__file__).resolve().parent

    # ──────────────────────────────────────────────────────────────────────
    # اختيار الـ checkpoint
    # ──────────────────────────────────────────────────────────────────────
    # finetune → نكمل من best.pt اللي عندنا (أسرع convergence وأعلى starting mAP)
    # scratch  → نبدأ من yolo11m.pt الرسمي (أفضل لو best.pt كان overfit أو على داتا مختلفة)
    model_file = "best.pt" if args.mode == "finetune" else "yolo11m.pt"
    model_path = resolve_path(model_file, script_dir)

    # مسار ملف الداتا YAML اللي فيه train/val paths وأسماء الـ classes
    data_yaml = resolve_path(args.data, script_dir)

    # ──────────────────────────────────────────────────────────────────────
    # التحقق من وجود الملفات قبل البدء
    # ──────────────────────────────────────────────────────────────────────
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Place the checkpoint in the training folder or provide a valid path."
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Place the YAML in the training folder or update the path."
        )

    # تحميل الموديل من الـ checkpoint المختار
    model = YOLO(str(model_path))

    # ──────────────────────────────────────────────────────────────────────
    # إعدادات التدريب (كل parameter موضح سببه)
    # ──────────────────────────────────────────────────────────────────────
    train_args = {
        # ── داتا ─────────────────────────────────────────────────────────
        "data": str(data_yaml),   # ملف الـ YAML بتفاصيل الداتا
        "single_cls": True,       # نعامل كل الـ classes كـ class واحدة "Swimmer"
                                  # مهم جداً لأن عندنا class واحدة بس

        # ── جدول التدريب ──────────────────────────────────────────────────
        "epochs": 80,             # عدد مرات التدريب على الداتا كاملة
        "imgsz": 640,             # حجم الصورة → توازن ممتاز بين الدقة والسرعة على RTX 3050
        "batch": 8,               # عدد الصور في كل batch → آمن لـ 4GB VRAM

        # ── الـ optimizer ─────────────────────────────────────────────────
        "optimizer": "SGD",       # SGD أفضل من Adam للـ generalization في التدريب الطويل
        "lr0": 0.003,             # learning rate ابتدائي → منخفض لأننا بنعمل fine-tune
        "momentum": 0.937,        # يساعد في تثبيت التحديثات وتسريع الـ convergence
        "weight_decay": 0.0005,   # L2 regularization → يقلل overfitting

        # ── جدول الـ learning rate ────────────────────────────────────────
        "cos_lr": True,           # Cosine annealing → يقلل الـ LR بشكل سلس ومنحنى
        "warmup_epochs": 3,       # يبدأ بـ LR صغير ويزيده تدريجياً لتفادي instability

        # ── منع الـ overfitting ───────────────────────────────────────────
        "patience": 25,           # يوقف التدريب لو مفيش تحسن لـ 25 epoch متتالية
        "close_mosaic": 15,       # يوقف الـ mosaic augmentation في آخر 15 epoch
                                  # عشان الموديل يستقر على detections واضحة

        # ── تعديلات الألوان والإضاءة ──────────────────────────────────────
        "hsv_h": 0.015,           # تغيير بسيط في الـ hue (اللون) ± 1.5%
        "hsv_s": 0.5,             # تغيير في الـ saturation ± 50% → محاكاة بيئات إضاءة مختلفة
        "hsv_v": 0.35,            # تغيير في الـ brightness ± 35% → مهم جداً للمسابح المختلفة

        # ── التحويلات الهندسية ────────────────────────────────────────────
        "degrees": 3.0,           # دوران بسيط ± 3° → السباحون دايماً أفقيين تقريباً
        "translate": 0.08,        # تحريك الصورة 8% → يحاكي تحرك السباح في الإطار
        "scale": 0.35,            # تغيير الحجم ± 35% → يعامل السباحين في مسافات مختلفة
        "shear": 0.5,             # ميل بسيط ± 0.5° → يتعامل مع كاميرات مش مستوية تماماً
        "perspective": 0.0003,    # تشويه perspective خفيف → يحاكي زوايا كاميرا مختلفة

        # ── القلب والتدوير ────────────────────────────────────────────────
        "fliplr": 0.5,            # قلب أفقي 50% → يضاعف الداتا (يسار = يمين)
        "flipud": 0.0,            # ❌ بدون قلب رأسي → سباح مقلوب مش منطقي

        # ── augmentations متقدمة ──────────────────────────────────────────
        "mosaic": 0.25,           # دمج 4 صور في صورة واحدة 25% → تنوع بدون مبالغة
        "mixup": 0.05,            # دمج صورتين معاً 5% → regularization خفيف
        "erasing": 0.1,           # حذف جزء عشوائي 10% → يحاكي إخفاء السباح بالمياه أو الحبال

        # ── إعدادات الهاردوير ─────────────────────────────────────────────
        "amp": True,              # Automatic Mixed Precision → يوفر VRAM ويسرع التدريب
        "workers": 4,             # عدد threads لتحميل الداتا (مناسب لـ quad-core)
        "device": 0,              # استخدام أول GPU (RTX 3050)

        # ── الحفظ والمخرجات ───────────────────────────────────────────────
        "save_period": 10,        # حفظ checkpoint كل 10 epochs (حماية من الـ crashes)
        "val": True,              # تقييم على الـ validation set بعد كل epoch
        "plots": True,            # رسم curves للـ loss وmAP والـ precision/recall
        "project": "runs/train",  # المجلد الرئيسي لحفظ نتائج التدريب
        "name": f"yolo_swimmer_{args.mode}_rtx3050",  # اسم مميز لكل run
        "exist_ok": True,         # لو المجلد موجود → يستخدمه بدون إنشاء مجلد جديد
    }

    # ──────────────────────────────────────────────────────────────────────
    # طباعة ملخص قبل البدء
    # ──────────────────────────────────────────────────────────────────────
    print("Starting YOLO training with the following setup:")
    print(f"  Mode:             {args.mode}")
    print(f"  Model checkpoint: {model_path}")
    print(f"  Dataset YAML:     {data_yaml}")
    print(f"  Epochs:           {train_args['epochs']}")
    print(f"  Image size:       {train_args['imgsz']}")
    print(f"  Batch size:       {train_args['batch']}")

    # ──────────────────────────────────────────────────────────────────────
    # بدء التدريب
    # ──────────────────────────────────────────────────────────────────────
    results = model.train(**train_args)

    # ──────────────────────────────────────────────────────────────────────
    # ملخص بعد التدريب
    # ──────────────────────────────────────────────────────────────────────
    # Ultralytics بيرجع save_dir → فيه كل النتائج والـ weights
    save_dir = Path(results.save_dir).resolve()
    best_model_path = (save_dir / "weights" / "best.pt").resolve()

    print("\nTraining complete. ✅")
    print(f"Results directory: {save_dir}")
    print(f"Best model path:   {best_model_path}")


if __name__ == "__main__":
    main()