if __name__ == "__main__":
    import torch
    from ultralytics import YOLO
    # Carregar modelo YOLOv8 pré-treinado
    model = YOLO("yolov8s-seg.pt")

    # Treinar usando dataset no formato COCO
    model.train(
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        device="0",          # Usa sua GPU
        project="runs/train",
        name="yolov8_workers",
        save=True,
        patience=20,         # Early stopping
        workers=4,           # Threads para carregar imagens
        val=True,            # Validação durante treino
    )

    # model.train(
    #     data="data.yaml",
    #     epochs=100,
    #     imgsz=640,
    #     batch=8,
    #     device="0",          # Usa sua GPU
    #     project="runs/train",
    #     name="yolov8_workers",
    #     save=True,
    #     patience=20,         # Early stopping
    #     workers=4,           # Threads para carregar imagens
    #     val=True,            # Validação durante treino
    # )