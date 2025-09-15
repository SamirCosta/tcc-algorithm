import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    images_info = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    for image_id, annotations in annotations_by_image.items():
        image_info = images_info[image_id]
        img_width = image_info['width']
        img_height = image_info['height']
        filename = image_info['file_name']
        
        txt_filename = Path(filename).stem + '.txt'
        txt_path = output_dir / txt_filename
        
        yolo_annotations = []
        
        for ann in annotations:
            class_id = categories[ann['category_id']]
            
            if 'segmentation' in ann and ann['segmentation']:
                segmentation = ann['segmentation'][0]
                normalized_seg = []
                for i in range(0, len(segmentation), 2):
                    x = segmentation[i] / img_width
                    y = segmentation[i + 1] / img_height
                    normalized_seg.extend([x, y])
                
                yolo_line = f"{class_id} " + " ".join(map(str, normalized_seg))
                yolo_annotations.append(yolo_line)
            
            elif 'bbox' in ann:
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                yolo_annotations.append(yolo_line)
        
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    print(f"Total de imagens processadas: {len(annotations_by_image)}")
    
    class_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: categories[x['id']])]
    with open(output_dir.parent / 'classes.txt', 'w') as f:
        f.write('\n'.join(class_names))
    
    return class_names

if __name__ == "__main__":
    convert_coco_to_yolo(
        coco_json_path="data/labels/train/instances_train.json",
        images_dir="data/images/train",
        output_dir="data/labels/train"
    )
    
    convert_coco_to_yolo(
        coco_json_path="data/labels/val/instances_val.json", 
        images_dir="data/images/val",
        output_dir="data/labels/val"
    )
