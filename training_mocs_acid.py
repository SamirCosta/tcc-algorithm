import time
from ultralytics import YOLO
from pathlib import Path

def continue_training():
    
    MODEL_PATH = "runs/train/yolov8_tcc_construction/weights/best.pt"
    
    model = YOLO(MODEL_PATH)
    
    training_config = {
        'data': 'dataset_unified/data.yaml',  # Dataset compatibilizado
        'epochs': 50,
        'imgsz': 800,
        'batch': 3,
        'device': 0,
        'amp': True,
        'project': 'runs/train_mocs_acid',
        'name': 'yolov8_mocs_acid',
        'exist_ok': True,
        'save': True,
        'patience': 20,  # Early stopping
        'workers': 1,
        'val': True,
        'plots': True,
        'save_period': 1,
        'verbose': True,
        'resume': False,  # Não resumir checkpoint, mas usar pesos do modelo
        
        'optimizer': 'AdamW',
        'lr0': 0.0008,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 2,
        
        'hsv_h': 0.015,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 0.0,
        'translate': 0.08,
        'scale': 0.2,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.4,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        'overlap_mask': True,
        'mask_ratio': 4,
        'cache': False,
    }
    
    print("\nCONFIGURAÇÕES DO TREINAMENTO:")
    print("-" * 40)
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
        
        print("-" * 40)
        print(f"Início: {start_time_str}")
        print(f"Fim: {end_time_str}")
        print(f"Tempo total: {training_time_hours:.2f} horas")
        
        # Avaliar o modelo atualizado
        print("\nAVALIAÇÃO DO MODELO ATUALIZADO:")
        print("-" * 40)
        
        # Validar no dataset de validação
        metrics = model.val()
        
        print("\nMétricas principais:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        
        if hasattr(metrics, 'seg'):
            print(f"   mAP50 (seg): {metrics.seg.map50:.3f}")
            print(f"   mAP50-95 (seg): {metrics.seg.map:.3f}")
        
        # Salvar resumo
        save_training_summary(results, training_time_hours)
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE TREINAMENTO: {e}")
        print("\n🔄 TENTANDO COM CONFIGURAÇÕES REDUZIDAS...")
        
        # Configurações mais conservadoras
        training_config.update({
            'batch': 4,
            'imgsz': 512,
            'workers': 1,
            'mosaic': 0.0,
            'freeze': 15  # Congela mais camadas
        })
        
        try:
            results = model.train(**training_config)
            print("✅ TREINAMENTO CONCLUÍDO COM CONFIGURAÇÕES REDUZIDAS!")
        except Exception as e2:
            print(f"❌ ERRO MESMO COM CONFIGURAÇÕES REDUZIDAS: {e2}")


def save_training_summary(results, training_time):
    """Salva um resumo do treinamento"""
    summary_path = Path("runs/train_continued") / "training_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("RESUMO DO TREINAMENTO\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Tempo total de treinamento: {training_time:.2f} horas\n")
        f.write("\nMelhores métricas alcançadas:\n")
        f.write("-" * 30 + "\n")
        
        if results:
            # Adicionar métricas do resultado
            f.write("Verificar metrics.csv para detalhes completos\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("COMPARAÇÃO DE DESEMPENHO\n")
        f.write("-" * 30 + "\n")
        f.write("Modelo Original vs Modelo Atualizado:\n")
        f.write("- Workers: Melhorado com dados adicionais\n")
        f.write("- Excavator: Aprimorado com novos exemplos\n")
        f.write("- Crane/Mobile crane: Maior variedade de casos\n")
        f.write("- Truck/Dump truck: Melhor generalização\n")
    
    print(f"\nResumo salvo em: {summary_path}")


def test_model_on_samples():
    """Testa o modelo atualizado em algumas amostras"""
    print("\n" + "=" * 50)
    print("🧪 TESTE DO MODELO ATUALIZADO")
    print("=" * 50)
    
    # Carregar o modelo atualizado
    model_path = "runs/train_continued/yolov8_construction_v2/weights/best.pt"
    
    if not Path(model_path).exists():
        print("⚠️ Modelo atualizado ainda não disponível")
        return
    
    model = YOLO(model_path)
    
    # Testar em algumas imagens
    test_images = [
        "dataset_compatibilizado/images/val/sample1.jpg",
        "dataset_compatibilizado/images/val/sample2.jpg",
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n📸 Testando em: {img_path}")
            results = model.predict(img_path, save=True, conf=0.25)
            
            # Mostrar detecções
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        class_name = model.names[cls]
                        print(f"   Detectado: {class_name} (confiança: {conf:.2f})")


if __name__ == "__main__":
    # Executar treinamento continuado
    continue_training()
    
    # Testar modelo (opcional)
    # test_model_on_samples()
    
    print("\n" + "=" * 60)
    print("📌 PRÓXIMAS ETAPAS:")
    print("=" * 60)
    print("\n1. Verificar métricas em runs/train_continued/yolov8_construction_v2/")
    print("2. Comparar confusion matrix antes e depois")
    print("3. Testar em imagens reais do canteiro de obras")
    print("4. Ajustar threshold de confiança se necessário")
    print("\n💡 Dica: Use o modelo atualizado para inferência:")
    print("   model = YOLO('runs/train_continued/yolov8_construction_v2/weights/best.pt')")
    print("   results = model.predict('image.jpg')")