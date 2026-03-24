from __future__ import annotations  # بيسمح باستخدام type hints بشكل متقدم (مش مهم قوي هنا)

import subprocess  # لتشغيل أوامر النظام (pip install)
import sys  # لمعرفة نسخة البايثون المستخدمة


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # بيحاول يستورد مكتبة YOLO
    except ImportError:
        # لو مش موجودة → ينزلها تلقائي
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])


def main() -> None:
    ensure_ultralytics()  # تأكد إن المكتبة موجودة
    from ultralytics import YOLO  # استيراد الموديل

    # 📁 مسار ملف الداتا (فيه paths + classes)
    data_yaml = r"C:\Swimming-analysis\ai\training\dataset.yaml"

    # 📁 موديل قديم هتعمل عليه fine-tune
    pretrained_run = r"C:\Swimming-analysis\ai\training\runs\train\yolo11m_swimmer_finetune_v2\weights\best.pt"

    # تحميل الموديل القديم
    model = YOLO(pretrained_run)

    # 🚀 بدء التدريب
    results = model.train(

        data=data_yaml,       # مكان الداتا
        imgsz=640,           # حجم الصور (توازن بين السرعة والدقة)
        epochs=50,           # عدد epochs (مرات التعلم)
        batch=8,             # عدد الصور في كل batch (حسب قوة GPU)
        device=0,            # استخدام كارت الشاشة

        # ======================
        # ⚙️ OPTIMIZER
        # ======================
        optimizer="SGD",     # طريقة تحديث الأوزان (ثابتة وموثوقة)
        lr0=0.005,           # learning rate (سرعة التعلم)
        momentum=0.937,      # يساعد في تثبيت التحديثات
        weight_decay=0.0005, # يقلل overfitting

        # ======================
        # 📉 LEARNING SCHEDULE
        # ======================
        cos_lr=True,         # learning rate يقل تدريجيًا (أفضل convergence)
        warmup_epochs=2,     # يبدأ ببطء في الأول لتفادي instability

        # ======================
        # 🛡️ REGULARIZATION
        # ======================
        patience=15,         # يقف لو مفيش تحسن بعد 15 epoch
        close_mosaic=10,     # يقفل mosaic في آخر epochs لتحسين الدقة

        # ======================
        # 🎨 AUGMENTATION
        # ======================

        # 🌈 تغيير الألوان والإضاءة
        hsv_h=0.02,          # تغيير بسيط في اللون (hue)
        hsv_s=0.5,           # تغيير في saturation (مهم للمياه)
        hsv_v=0.4,           # 🔥 brightness variation (مهم جدًا)

        # 🔄 حركة وتحويلات
        degrees=5.0,         # دوران بسيط للصورة
        translate=0.1,       # تحريك الصورة (simulate camera shift)
        scale=0.4,           # zoom in/out (مهم جدًا للـ distance)
        shear=1.0,           # ميل بسيط (زاوية)

        # 🎥 perspective
        perspective=0.0005,  # يحاكي الكاميرا وهي مش مستقيمة

        # 🔁 flipping
        fliplr=0.5,          # يقلب الصورة يمين/شمال
        flipud=0.0,          # ❌ متقلبش فوق/تحت (مش منطقي للسباح)

        # 🧩 mix augmentations
        mosaic=0.3,          # دمج 4 صور في صورة واحدة (تنوع عالي)
        mixup=0.1,           # دمج صورتين (يساعد مع clutter)

        # 🌊 noise / occlusion
        erasing=0.15,        # يحذف جزء من الصورة (simulate splash/occlusion)

        # ======================
        # ⚙️ SYSTEM
        # ======================
        workers=4,           # عدد threads لتحميل البيانات
        amp=True,            # تسريع باستخدام mixed precision
        seed=42,             # لتكرار نفس النتائج
        deterministic=False, # يسمح ببعض العشوائية (أفضل للتعلم)

        # ======================
        # 💾 OUTPUT
        # ======================
        project=r"C:\Swimming-analysis\ai\training\runs",  # مكان حفظ النتائج
        name="yolo11m_swimmer_augmented_v5",               # اسم التجربة

        val=True,            # يعمل validation أثناء التدريب
        save=True,           # يحفظ الموديل
        plots=True,          # يرسم graphs (loss / mAP)
    )

    print("✅ Training complete.")
    print("Best model:", model.trainer.best)  # أفضل وزن طلع
    print("Results folder:", model.trainer.save_dir)  # مكان النتائج


if __name__ == "__main__":
    main()