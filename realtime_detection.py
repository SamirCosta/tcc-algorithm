import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
from pathlib import Path

class ConstructionObjectDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        self.colors = {
            0: (255, 0, 0),     # Worker - Azul
            1: (0, 255, 0),     # Static crane - Verde
            2: (0, 0, 255),     # Hanging head - Vermelho
            3: (255, 255, 0),   # Crane - Ciano
            4: (255, 0, 255),   # Roller - Magenta
            5: (0, 255, 255),   # Bulldozer - Amarelo
            6: (128, 0, 128),   # Excavator - Roxo
            7: (255, 165, 0),   # Truck - Laranja
            8: (0, 128, 0),     # Loader - Verde escuro
            9: (128, 128, 0),   # Pump truck - Oliva
            10: (0, 128, 128),  # Concrete mixer - Teal
            11: (128, 0, 0),    # Pile driving - Marrom
            12: (192, 192, 192) # Other vehicle - Prata
        }
        
        self.class_names = {
            0: "Worker", 1: "Static crane", 2: "Hanging head", 3: "Crane",
            4: "Roller", 5: "Bulldozer", 6: "Excavator", 7: "Truck",
            8: "Loader", 9: "Pump truck", 10: "Concrete mixer", 
            11: "Pile driving", 12: "Other vehicle"
        }
    
    def draw_detections(self, frame, results):
        detections_count = {name: 0 for name in self.class_names.values()}
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extrair informações da detecção
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filtrar por confiança
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Contar detecções
                    class_name = self.class_names.get(class_id, f"Class_{class_id}")
                    detections_count[class_name] += 1
                    
                    # Cor da classe
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # Desenhar bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Texto da detecção
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Fundo do texto
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Texto
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Desenhar máscara de segmentação se disponível
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                if i < len(results[0].boxes):
                    class_id = int(results[0].boxes[i].cls[0].cpu().numpy())
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # Redimensionar máscara para o tamanho do frame
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_colored = np.zeros_like(frame)
                    mask_colored[mask_resized > 0.5] = color
                    
                    # Aplicar máscara com transparência
                    frame = cv2.addWeighted(frame, 0.8, mask_colored, 0.2, 0)
        
        return frame, detections_count
    
    def draw_info_panel(self, frame, detections_count, fps):
        panel_height = 200
        panel_width = 300
        
        # Criar painel semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Título
        cv2.putText(frame, "DETECCOES EM TEMPO REAL", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Contadores de detecção
        y_offset = 80
        total_objects = 0
        for class_name, count in detections_count.items():
            if count > 0:
                total_objects += count
                cv2.putText(frame, f"{class_name}: {count}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        # Total de objetos
        cv2.putText(frame, f"Total: {total_objects}", (20, y_offset + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_source, output_path=None, show_display=True):
        # Abrir fonte de vídeo
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir {video_source}")
            return
        
        # Propriedades do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Vídeo: {width}x{height} @ {fps}fps")
        
        # Configurar gravação se necessário
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Variáveis para FPS
        frame_count = 0
        start_time = time.time()
        fps_display = 0
        
        print("Pressione 'q' para sair, 'p' para pausar/despausar")
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Fim do vídeo ou erro na leitura")
                        break
                    
                    # Fazer predição
                    results = self.model(frame, verbose=False)
                    
                    # Desenhar detecções
                    frame_with_detections, detections_count = self.draw_detections(frame, results)
                    
                    # Calcular FPS
                    frame_count += 1
                    if frame_count % 30 == 0:  # Atualizar FPS a cada 30 frames
                        elapsed_time = time.time() - start_time
                        fps_display = 30 / elapsed_time
                        start_time = time.time()
                    
                    # Desenhar painel de informações
                    final_frame = self.draw_info_panel(frame_with_detections, 
                                                     detections_count, fps_display)
                    
                    # Salvar frame se necessário
                    if out:
                        out.write(final_frame)
                    
                    # Mostrar frame
                    if show_display:
                        cv2.imshow('Detecção de Objetos - Construção Civil', final_frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Pausado" if paused else "Retomado")
        
        except KeyboardInterrupt:
            print("\nParando detecção...")
        
        finally:
            # Limpeza
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"Processamento concluído. Total de frames: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Detecção em tempo real para construção civil')
    parser.add_argument('--model', required=True, help='Caminho para o modelo treinado (.pt)')
    parser.add_argument('--source', default=0, help='Fonte do vídeo (caminho ou 0 para webcam)')
    parser.add_argument('--output', help='Caminho para salvar vídeo processado')
    parser.add_argument('--confidence', type=float, default=0.5, help='Limite de confiança')
    parser.add_argument('--no-display', action='store_true', help='Não mostrar vídeo na tela')
    
    args = parser.parse_args()
    
    # Verificar se o modelo existe
    if not Path(args.model).exists():
        print(f"Erro: Modelo não encontrado em {args.model}")
        return
    
    # Converter source para int se for número
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
        if not Path(video_source).exists():
            print(f"Erro: Vídeo não encontrado em {video_source}")
            return
    
    # Criar detector
    detector = ConstructionObjectDetector(args.model, args.confidence)
    
    # Processar vídeo
    detector.process_video(
        video_source=video_source,
        output_path=args.output,
        show_display=not args.no_display
    )

if __name__ == "__main__":
    main()