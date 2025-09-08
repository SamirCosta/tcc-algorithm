if __name__ == "__main__":
    import time
    from ultralytics import YOLO
    
    model = YOLO("yolov8s-seg.pt")

    training_config = {
        'data': 'data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': 0,
        'project': 'runs/train',
        'name': 'yolov8_tcc_construction',
        'save': True,
        'patience': 20,
        'workers': 4,
        'val': True,
        'plots': True,
        'save_period': 1,
        'verbose': True,

        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,

        'overlap_mask': True,
        'mask_ratio': 4,
    }

    print("CONFIGURAÇÕES:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")

    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S", time.localtime(start_time))
    print(f"\nInício do treinamento: {start_time_str}")

    try:
        results = model.train(**training_config)

        end_time = time.time()
        training_time_hours = (end_time - start_time) / 3600
        end_time_str = time.strftime("%H:%M:%S", time.localtime(end_time))

        print("TREINAMENTO CONCLUÍDO COM SUCESSO")
        print(f"Início: {start_time_str}")
        print(f"Fim: {end_time_str}")
        print(f"Tempo total: {training_time_hours:.2f} horas")

    except Exception as e:
        print(f"\nERRO DURANTE TREINAMENTO: {e}")
        print("\nTENTANDO COM CONFIGURAÇÕES REDUZIDAS...")

        training_config.update({
            'batch': 8,
            'imgsz': 512,
            'workers': 1,
            'amp': False
        })

        try:
            results = model.train(**training_config)
            print("TREINAMENTO CONCLUÍDO COM CONFIGURAÇÕES REDUZIDAS!")
        except Exception as e2:
            print(f"ERRO MESMO COM CONFIGURAÇÕES REDUZIDAS: {e2}")
            print("Tente reiniciar o runtime e reduzir ainda mais as configurações")