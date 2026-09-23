from ultralytics import YOLO

dataset = "util/yolo_dataset/dataset.yaml"

def validate():
    # Load your fine-tuned model
    model = YOLO("out")

    # Run validation on the dataset defined in dataset.yaml
    metrics = model.val(
        data=dataset,
        split="val",      # Explicitly evaluate on the val split
        imgsz=640,
        batch=16,
        project="couch_detector",
        name="validation_results"
    )

    # Print overall performance metrics
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


def main():
    # Load the pre-trained YOLO11 small model
    model = YOLO('yolo11s.pt')

    results = model.train(
        data=dataset,
        epochs=150,
        imgsz=640,
        batch=16,
        patience=20,
        save=True,

        # --- Loss & Balance Tuning (FP Optimization) ---
        cls=1.5,  # Increased further since we can't use focal loss. Heavily penalizes False Positives.
        box=8.5,  # Tightens bounding box overlap
        weight_decay=0.001,

        # --- Top-Down Augmentations ---
        scale=0.5,
        translate=0.1,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,  # Crucial for learning complex backgrounds to ignore
        hsv_s=0.2,
        hsv_v=0.2,

        # --- System ---
        device=0,
        workers=8,
        project="couch_detector",
        name="yolo11s_tuned",
    )

    print(f"Training complete!")


if __name__ == '__main__':
    main()