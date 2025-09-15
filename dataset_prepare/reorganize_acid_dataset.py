import json
import os
import shutil
from pathlib import Path
import random
from collections import defaultdict
import yaml

class ACIDDatasetReorganizer:
    def __init__(self, acid_path, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.acid_path = Path(acid_path)
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "As proporções devem somar 1.0"

        print("Configuração do Reorganizador:")
        print(f"Dataset ACID: {self.acid_path}")
        print(f"Saída: {self.output_path}")
        print(f"Divisão: {train_ratio*100:.0f}% treino, {val_ratio*100:.0f}% validação, {test_ratio*100:.0f}% teste")

    def reorganize(self):
        print("\n" + "="*60)
        print("INICIANDO REORGANIZAÇÃO DO DATASET ACID")
        print("="*60)

        if not self._verify_source_files():
            return False

        self._create_directory_structure()
        annotations_data = self._load_annotations()
        splits = self._split_dataset_stratified(annotations_data)
        self._process_splits(annotations_data, splits)
        self._create_yaml_config()
        self._generate_report(splits)

        print("\nREORGANIZAÇÃO CONCLUÍDA COM SUCESSO!")

    def _verify_source_files(self):
        print("\nVerificando arquivos de origem...")

        instances_path = self.acid_path / "instances.json"
        images_path = self.acid_path / "images"

        if not instances_path.exists():
            print(f"Arquivo não encontrado: {instances_path}")
            return False

        if not images_path.exists():
            print(f"Pasta de imagens não encontrada: {images_path}")
            return False

        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        print(f"Encontradas {len(image_files)} imagens")

        return True

    def _create_directory_structure(self):
        print("\nCriando estrutura de diretórios...")

        dirs = [
            self.output_path / "images" / "train",
            self.output_path / "images" / "val",
            self.output_path / "images" / "test",
            self.output_path / "labels" / "train",
            self.output_path / "labels" / "val",
            self.output_path / "labels" / "test",
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   {dir_path.relative_to(self.output_path)}")

    def _load_annotations(self):
        print("\nCarregando anotações...")

        instances_path = self.acid_path / "instances.json"
        with open(instances_path, 'r') as f:
            data = json.load(f)

        print(f"   {len(data['images'])} imagens")
        print(f"   {len(data['categories'])} categorias")
        print(f"   {len(data['annotations'])} anotações")

        print("\nCategorias encontradas:")
        for cat in data['categories']:
            ann_count = sum(1 for ann in data['annotations'] if ann['category_id'] == cat['id'])
            print(f"      - {cat['name']:20} (ID: {cat['id']:2}) - {ann_count:4} anotações")

        return data

    def _split_dataset_stratified(self, data):
        print("\nDividindo dataset de forma estratificada...")

        images_by_category = defaultdict(set)
        image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        annotations_by_image = defaultdict(list)
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann['category_id'])

        for img_id, categories in annotations_by_image.items():
            main_category = max(set(categories), key=categories.count)
            images_by_category[main_category].add(img_id)

        train_images = set()
        val_images = set()
        test_images = set()

        for category_id, image_ids in images_by_category.items():
            image_list = list(image_ids)
            random.shuffle(image_list)

            n_total = len(image_list)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            train_images.update(image_list[:n_train])
            val_images.update(image_list[n_train:n_train + n_val])
            test_images.update(image_list[n_train + n_val:])

        all_image_ids = set(img['id'] for img in data['images'])
        unannotated = all_image_ids - (train_images | val_images | test_images)
        if unannotated:
            print(f"{len(unannotated)} imagens sem anotações adicionadas ao treino")
            train_images.update(unannotated)

        splits = {
            'train': sorted(list(train_images)),
            'val': sorted(list(val_images)),
            'test': sorted(list(test_images))
        }

        print(f"Train: {len(splits['train'])} imagens")
        print(f"Val:   {len(splits['val'])} imagens")
        print(f"Test:  {len(splits['test'])} imagens")

        return splits

    def _process_splits(self, data, splits):
        print("\nProcessando splits...")

        image_id_to_data = {img['id']: img for img in data['images']}
        annotations_by_image = defaultdict(list)
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)

        for split_name, image_ids in splits.items():
            print(f"\nProcessando {split_name}...")

            split_data = {
                "images": [],
                "categories": data['categories'],
                "annotations": []
            }

            annotation_id = 1
            images_copied = 0
            images_not_found = 0

            for img_id in image_ids:
                if img_id not in image_id_to_data:
                    continue

                img_data = image_id_to_data[img_id]
                src_image = self.acid_path / "images" / img_data['file_name']
                dst_image = self.output_path / "images" / split_name / img_data['file_name']

                if src_image.exists():
                    shutil.copy2(src_image, dst_image)
                    images_copied += 1

                    split_data['images'].append(img_data)

                    for ann in annotations_by_image[img_id]:
                        new_ann = ann.copy()
                        new_ann['id'] = annotation_id
                        split_data['annotations'].append(new_ann)
                        annotation_id += 1
                else:
                    images_not_found += 1
                    print(f"Imagem não encontrada: {img_data['file_name']}")

            output_json = self.output_path / "labels" / split_name / f"instances_{split_name}.json"
            with open(output_json, 'w') as f:
                json.dump(split_data, f, indent=2)

            print(f"{images_copied} imagens copiadas")
            if images_not_found > 0:
                print(f"{images_not_found} imagens não encontradas")
            print(f"{len(split_data['annotations'])} anotações salvas")

    def _create_yaml_config(self):
        print("\nCriando arquivo de configuração YAML...")

        instances_path = self.acid_path / "instances.json"
        with open(instances_path, 'r') as f:
            data = json.load(f)

        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'train_annotations': 'labels/train/instances_train.json',
            'val_annotations': 'labels/val/instances_val.json',
            'test_annotations': 'labels/test/instances_test.json',
            'names': {},
            'nc': len(data['categories'])
        }

        for cat in data['categories']:
            yolo_id = cat['id'] - 1 if cat['id'] > 0 else 0
            config['names'][yolo_id] = cat['name']

        yaml_path = self.output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Configuração salva em: {yaml_path}")

    def _generate_report(self, splits):
        print("\nGerando relatório...")

        report_path = self.output_path / "reorganization_report.txt"
        instances_path = self.acid_path / "instances.json"
        with open(instances_path, 'r') as f:
            data = json.load(f)

        annotations_by_image = defaultdict(list)
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)

        with open(report_path, 'w') as f:
            f.write("RELATÓRIO DE REORGANIZAÇÃO DO DATASET ACID\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Dataset original: {self.acid_path}\n")
            f.write(f"Dataset reorganizado: {self.output_path}\n")
            f.write(f"Data: {Path.cwd()}\n\n")

            f.write("DIVISÃO DO DATASET\n")
            f.write("-" * 30 + "\n")

            total_images = len(data['images'])
            for split_name, image_ids in splits.items():
                percentage = (len(image_ids) / total_images) * 100
                f.write(f"{split_name:10} {len(image_ids):5} imagens ({percentage:5.1f}%)\n")

            f.write("\n" + "-" * 30 + "\n")
            f.write("DISTRIBUIÇÃO DE CLASSES POR SPLIT\n")
            f.write("-" * 30 + "\n")

            for split_name, image_ids in splits.items():
                f.write(f"\n{split_name.upper()}:\n")

                class_counts = defaultdict(int)
                for img_id in image_ids:
                    for ann in annotations_by_image[img_id]:
                        class_counts[ann['category_id']] += 1

                id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

                for class_id in sorted(class_counts.keys()):
                    class_name = id_to_name.get(class_id, f"Unknown_{class_id}")
                    count = class_counts[class_id]
                    f.write(f"  {class_name:20} {count:5} anotações\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("REORGANIZAÇÃO CONCLUÍDA COM SUCESSO!\n")

        print(f"Relatório salvo em: {report_path}")


def main():
    print("REORGANIZADOR DE DATASET ACID PARA FORMATO COCO/YOLO")
    print("=" * 60)

    ACID_PATH = r"C:\Users\samir\git\tcc-algorithm\ACID"
    OUTPUT_PATH = r"C:\Users\samir\git\tcc-algorithm\ACID_reorganized"

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    print("\nConfigurações:")
    print(f"   Origem: {ACID_PATH}")
    print(f"   Destino: {OUTPUT_PATH}")
    print(f"   Divisão: {TRAIN_RATIO*100:.0f}% treino, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% teste")

    response = input("\nDeseja continuar? (s/n): ")
    if response.lower() != 's':
        print("Operação cancelada")
        return

    reorganizer = ACIDDatasetReorganizer(
        acid_path=ACID_PATH,
        output_path=OUTPUT_PATH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

    reorganizer.reorganize()


if __name__ == "__main__":
    main()
