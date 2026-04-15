from ultralytics import YOLO
import os
import cv2
from ultralytics.utils.loss import v8DetectionLoss
import torch
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou

def apply_clahe_offline(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # 의료 영상이므로 흑백으로 로드
        
        if img is not None:
            # CLAHE 전처리 적용
            enhanced_img = clahe.apply(img)
            
            # 전처리된 이미지 저장
            cv2.imwrite(os.path.join(output_folder, img_name), enhanced_img)

# apply_clahe_offline('./data/ARCADE/stenosis/train/images/', './data/ARCADE/stenosis_yolo/images/train/')
# apply_clahe_offline('./data/ARCADE/stenosis/val/images/', './data/ARCADE/stenosis_yolo/images/val/')
# apply_clahe_offline('./data/ARCADE/stenosis/test/images/', './data/ARCADE/stenosis_yolo/images/test/')



def train_yolo_standard():
    model = YOLO("yolov8m.pt") 
    

    print("학습 시작")
    results = model.train(
        data="./data/ARCADE/stenosis_yolo/dataset.yaml",   
        optimizer="AdamW",   
        lr0=0.001,           
        weight_decay=0.0005,  
        epochs=250,              
        imgsz=512,               
        batch=16,                
        device=0,               
        project="C:/Code/workspace/segmentation_framework/ARCADE_postprocessing",  
        name="stenosis_detect",   
        mosaic=0.0,     
        mixup=0.0,       
        degrees=15.0,    
        translate=0.1,   
        scale=0.1,       
        hsv_h=0.0,
        fliplr=0.5,      # 좌우 반전 50% 적용
        flipud=0.5,

    )

if __name__ == '__main__':
    train_yolo_standard()