if __name__ == "__main__":
    import time
    from ultralytics import YOLO
    
    model = YOLO("yolov8s-seg.pt")

    training_config = {
        'data': 'data.yaml',
        'epochs': 150,              # Mais épocas para melhor convergência
        'imgsz': 640,               # Tamanho de imagem otimizado
        'batch': 3,                 # Batch size para RTX 2080 com YOLOv8l
        'device': 0,                # GPU 0
        'project': 'runs/train',
        'name': 'yolov8s',
        'save': True,
        'save_period': 5,           # Salvar checkpoint a cada 5 épocas
        'patience': 30,             # Early stopping patience
        'workers': 4,               # Data loading workers
        'exist_ok': True,           # Sobrescrever pasta se existir
        'pretrained': True,         # Usar pesos pré-treinados
        'verbose': True,            # Saída detalhada
        'seed': 42,                 # Seed para reprodutibilidade
        'deterministic': False,     # False para melhor performance
        'single_cls': False,        # Multi-class
        'rect': False,              # Rectangular training
        'cos_lr': True,             # Cosine learning rate scheduler
        'close_mosaic': 10,         # Desativar mosaic nas últimas 10 épocas
        'resume': False,            # Não resumir treinamento anterior
        'amp': True,                # Automatic Mixed Precision (economiza VRAM)
        'fraction': 1.0,            # Usar dataset completo
        'profile': False,           # Não fazer profiling
        'freeze': None,             # Não congelar camadas
        
        # Otimização
        'optimizer': 'AdamW',       # AdamW optimizer
        'lr0': 0.002,              # Learning rate inicial
        'lrf': 0.01,               # Learning rate final (lr0 * lrf)
        'momentum': 0.937,          # Momentum do SGD / beta1 do Adam
        'weight_decay': 0.0005,     # Weight decay
        'warmup_epochs': 5,         # Épocas de warmup
        'warmup_momentum': 0.8,     # Momentum inicial do warmup
        'warmup_bias_lr': 0.1,      # Bias learning rate do warmup
        
        # Pesos das losses
        'box': 7.5,                 # Box loss gain
        'cls': 1.0,                 # Classification loss gain
        'dfl': 1.5,                 # Distribution focal loss gain
        
        # Data Augmentation
        'hsv_h': 0.015,            # Variação HSV-Hue
        'hsv_s': 0.7,              # Variação HSV-Saturation
        'hsv_v': 0.4,              # Variação HSV-Value
        'degrees': 5.0,            # Rotação (+/- graus)
        'translate': 0.1,          # Translação (+/- fração)
        'scale': 0.5,              # Escala (+/- gain)
        'shear': 2.0,              # Cisalhamento (+/- graus)
        'perspective': 0.0001,     # Perspectiva (+/- fração)
        'flipud': 0.0,             # Flip vertical (probabilidade)
        'fliplr': 0.5,             # Flip horizontal (probabilidade)
        'bgr': 0.0,                # Flip BGR channels (probabilidade)
        'mosaic': 1.0,             # Mosaic (probabilidade)
        'mixup': 0.1,              # Mixup (probabilidade)
        'copy_paste': 0.1,         # Copy-paste segmentation (probabilidade)
        'auto_augment': 'randaugment',  # Auto augmentation policy
        'erasing': 0.0,            # Random erasing (probabilidade)
        'crop_fraction': 1.0,      # Crop classification images
        
        # Segmentação específica
        'overlap_mask': True,       # Máscaras podem sobrepor durante treinamento
        'mask_ratio': 4,           # Downsampling ratio de máscara
        
        # Hiperparâmetros adicionais
        'dropout': 0.0,            # Dropout (somente para classificação)
        'val': True,               # Validar durante treinamento
        'plots': True,             # Criar plots
        'cache': True,             # Cache de imagens (True/ram, disk ou False)
        'iou': 0.7,                # IoU threshold para NMS no treinamento
        'max_det': 300,            # Máximo de detecções por imagem
        'vid_stride': 1,           # Video frame-rate stride
        'line_width': 1,           # Largura da linha das bounding boxes
        'visualize': False,        # Visualizar features
        'augment': False,          # Augmentation na predição
        'agnostic_nms': False,     # NMS class-agnostic
        'retina_masks': False,     # Usar máscaras de alta resolução
        'show': False,             # Mostrar resultados
        'save_frames': False,      # Salvar frames de predição
        'save_txt': False,         # Salvar resultados em txt
        'save_conf': False,        # Salvar confidences nos txts
        'save_crop': False,        # Salvar crops de predição
        'show_labels': True,       # Mostrar labels nas visualizações
        'show_conf': True,         # Mostrar confidences
        'show_boxes': True,        # Mostrar boxes nas segmentações
        'keras': False,            # Usar Keras
    }

    print("CONFIGURAÇÕES:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")

    start_time = time.time()
    start_time_str = time.strftime("%H:%M:%S", time.localtime(start_time))
    print(f"\nInício do treinamento: {start_time_str}")

    results = model.train(**training_config)

    end_time = time.time()
    training_time_hours = (end_time - start_time) / 3600
    end_time_str = time.strftime("%H:%M:%S", time.localtime(end_time))

    print("TREINAMENTO CONCLUÍDO COM SUCESSO")
    print(f"Início: {start_time_str}")
    print(f"Fim: {end_time_str}")
    print(f"Tempo total: {training_time_hours:.2f} horas")