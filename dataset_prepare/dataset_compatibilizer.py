import json
import os
import shutil
from pathlib import Path
import yaml

class DatasetCompatibilizer:
    def __init__(self, old_dataset_path, new_dataset_path, output_path):
        self.old_dataset_path = Path(old_dataset_path)
        self.new_dataset_path = Path(new_dataset_path)
        self.output_path = Path(output_path)
        
        self.class_mapping = {
            'excavator': 'Excavator',
            'mobile_crane': 'Crane',
            'tower_crane': 'Static crane',
            'wheel_loader': 'Loader',
            'dozer': 'Bulldozer',
            'concrete_mixer_truck': 'Concrete mixer',
            'cement_truck': 'Pump truck',
            'dump_truck': 'Truck',
            'compactor': 'Roller',
            'backhoe_loader': 'Loader',
            'grader': 'Other vehicle',
        }
        
        self.old_class_ids = {
            'Worker': 0,
            'Static crane': 1,
            'Hanging head': 2,
            'Crane': 3,
            'Roller': 4,
            'Bulldozer': 5,
            'Excavator': 6,
            'Truck': 7,
            'Loader': 8,
            'Pump truck': 9,
            'Concrete mixer': 10,
            'Pile driving': 11,
            'Other vehicle': 12
        }
        
    def create_unified_dataset(self):
        self._create_directory_structure()
        self._process_new_annotations()
        self._copy_images()
        self._create_config_yaml()
        
    def _create_directory_structure(self):
        dirs = [
            self.output_path / 'images' / 'train',
            self.output_path / 'images' / 'val',
            self.output_path / 'images' / 'test',
            self.output_path / 'labels' / 'train',
            self.output_path / 'labels' / 'val',
            self.output_path / 'labels' / 'test'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _process_new_annotations(self):
        splits = ['train', 'val', 'test']
        
        for split in splits:
            json_path = self.new_dataset_path / 'labels' / split / f'instances_{split}.json'
            if not json_path.exists():
                print(f"Arquivo não encontrado: {json_path}")
                continue
            
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            
            new_cat_id_to_name = {}
            for cat in coco_data['categories']:
                new_cat_id_to_name[cat['id']] = cat['name']
            
            image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
            image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
            
            annotations_by_image = {}
            for ann in coco_data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)
            
            for img_id, annotations in annotations_by_image.items():
                if img_id not in image_id_to_filename:
                    continue
                    
                filename = image_id_to_filename[img_id]
                img_width, img_height = image_id_to_size[img_id]
                
                label_filename = Path(filename).stem + '.txt'
                label_path = self.output_path / 'labels' / split / label_filename
                
                yolo_labels = []
                for ann in annotations:
                    
                    new_class_name = new_cat_id_to_name.get(ann['category_id'])
                    if not new_class_name:
                        continue
                    
                    old_class_name = self.class_mapping.get(new_class_name)
                    if not old_class_name:
                        print(f"Classe '{new_class_name}' não tem mapeamento")
                        continue
                    
                    old_class_id = self.old_class_ids[old_class_name]
                    
                    if 'segmentation' in ann and ann['segmentation']:
                        segments = ann['segmentation']
                        if isinstance(segments, list) and len(segments) > 0:
                            segment = segments[0]
                            
                            normalized_segments = []
                            for i in range(0, len(segment), 2):
                                x = segment[i] / img_width
                                y = segment[i + 1] / img_height
                                normalized_segments.extend([x, y])
                            
                            yolo_line = f"{old_class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_segments])
                            yolo_labels.append(yolo_line)
                    
                    elif 'bbox' in ann:
                        
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        yolo_line = f"{old_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        yolo_labels.append(yolo_line)
                
                if yolo_labels:
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_labels))
            
            print(f"{split}: {len(annotations_by_image)} imagens processadas")
    
    def _copy_images(self):
        splits = ['train', 'val', 'test']
        
        for split in splits:
            src_dir = self.new_dataset_path / 'images' / split
            dst_dir = self.output_path / 'images' / split
            
            if not src_dir.exists():
                print(f"Diretório não encontrado: {src_dir}")
                continue
            
            print(f"Copiando imagens de {split}...")
            
            for img_file in src_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img_file, dst_dir / img_file.name)
            
            num_images = len(list(dst_dir.glob('*')))
            print(f"{split}: {num_images} imagens copiadas")
    
    def _create_config_yaml(self):
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'Worker',
                1: 'Static crane',
                2: 'Hanging head',
                3: 'Crane',
                4: 'Roller',
                5: 'Bulldozer',
                6: 'Excavator',
                7: 'Truck',
                8: 'Loader',
                9: 'Pump truck',
                10: 'Concrete mixer',
                11: 'Pile driving',
                12: 'Other vehicle'
            },
            'nc': 13
        }
        
        yaml_path = self.output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Arquivo de configuração criado: {yaml_path}")
        self._save_mapping_report()
    
    def _save_mapping_report(self):
        report_path = self.output_path / 'class_mapping_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("RELATÓRIO DE MAPEAMENTO DE CLASSES\n")
            f.write("=" * 50 + "\n\n")
            f.write("Mapeamento aplicado:\n")
            f.write("-" * 30 + "\n")
            
            for new_class, old_class in sorted(self.class_mapping.items()):
                old_id = self.old_class_ids[old_class]
                f.write(f"{new_class:25} -> {old_class:20} (ID: {old_id})\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Classes sem dados novos (mantidas apenas do modelo original):\n")
            f.write("-" * 30 + "\n")
            
            mapped_old_classes = set(self.class_mapping.values())
            for old_class, old_id in self.old_class_ids.items():
                if old_class not in mapped_old_classes:
                    f.write(f"- {old_class} (ID: {old_id})\n")
        
        print(f"Relatório de mapeamento salvo: {report_path}")


def main():
    OLD_DATASET_PATH = "C:/Users/samir/git/tcc-algorithm"
    NEW_DATASET_PATH = "C:/Users/samir/git/tcc-algorithm/ACID_reorganized"
    OUTPUT_PATH = "C:/Users/samir/git/tcc-algorithm/dataset_unified"
    
    compatibilizer = DatasetCompatibilizer(
        OLD_DATASET_PATH,
        NEW_DATASET_PATH,
        OUTPUT_PATH
    )
    
    compatibilizer.create_unified_dataset()

if __name__ == "__main__":
    main()
