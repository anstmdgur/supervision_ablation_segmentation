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

original_forward = BboxLoss.forward

def nwd_bbox_forward_wrapper(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
    # 1. 기존 YOLO의 기본 Loss(CIoU, DFL)를 에러 없이 원본 그대로 먼저 계산합니다.
    loss_box, loss_dfl = original_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
    
    # 2. 정답(Positive) 샘플이 있는 경우에만 NWD를 추가 계산하여 혼합합니다.
    if fg_mask.sum() > 0:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        p_boxes = pred_bboxes[fg_mask]
        t_boxes = target_bboxes[fg_mask]

        # 중심점 및 너비/높이 계산
        p_cx = (p_boxes[:, 0] + p_boxes[:, 2]) / 2
        p_cy = (p_boxes[:, 1] + p_boxes[:, 3]) / 2
        p_w  = p_boxes[:, 2] - p_boxes[:, 0]
        p_h  = p_boxes[:, 3] - p_boxes[:, 1]

        t_cx = (t_boxes[:, 0] + t_boxes[:, 2]) / 2
        t_cy = (t_boxes[:, 1] + t_boxes[:, 3]) / 2
        t_w  = t_boxes[:, 2] - t_boxes[:, 0]
        t_h  = t_boxes[:, 3] - t_boxes[:, 1]

        # Wasserstein Distance 제곱
        w2_sq = (p_cx - t_cx)**2 + (p_cy - t_cy)**2 + ((p_w - t_w) / 2)**2 + ((p_h - t_h) / 2)**2
        
        # NWD 계산 (상수 C=12.8)
        C = 12.8 
        nwd = torch.exp(-torch.sqrt(w2_sq + 1e-10) / C)

        # 차원을 맞추고 NWD Loss 스케일링
        loss_nwd = (1.0 - nwd).unsqueeze(-1)
        nwd_loss_sum = (loss_nwd * weight).sum() / target_scores_sum

        # 기존 CIoU 손실(loss_box)과 NWD 손실을 1:1로 부드럽게 결합
        loss_box = 0.5 * loss_box + 0.5 * nwd_loss_sum

    return loss_box, loss_dfl

# 함수 덮어쓰기
BboxLoss.forward = nwd_bbox_forward_wrapper

def train_yolo_standard():
    model = YOLO("yolov8m.pt") 
    

    print("학습 시작")
    results = model.train(
        data="./data/ARCADE/stenosis_yolo/dataset.yaml",   
        optimizer="AdamW",   
        lr0=0.001,           
        weight_decay=0.0005,  
        epochs=100,              
        imgsz=384,               
        batch=32,                
        device=0,               
        project="C:/Code/workspace/segmentation_framework/ARCADE_yolo",  
        name="stenosis_detect",   
        mosaic=0.0,     
        mixup=0.0,       
        degrees=15.0,    
        translate=0.1,   
        scale=0.1,       
        hsv_h=0.0,
        fliplr=0.5,      # 좌우 반전 50% 적용
        flipud=0.5,

        box=7.5,         # Bounding Box Loss 가중치를 7.5로 높여 NWD의 영향력을 극대화
    )

    print(f"가장 좋은 모델 가중치는 yolo_results/stenosis_detect/weights/best.pt 에 저장되었습니다.")

if __name__ == '__main__':
    train_yolo_standard()