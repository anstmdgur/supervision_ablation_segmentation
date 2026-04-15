import dataset
import model
import train
import torch
import torch.optim
import random
import os
import numpy as np
import yaml
import pandas as pd
import time

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def main_train(config_name, trainable=True, test = True, load = False):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f'./segmentation_framework/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = config[config_name]
    path = config['path']
    os.makedirs(path, exist_ok=True)



    early_stopping = train.EarlyStopping(patience = 25, delta = 0.00005, path = path)
    mymodel = model.select_model(config).to(device)

    if load:
        checkpoint = torch.load(os.path.join(f"./segmentation_framework/{load}/checkpoint_fine_tune.pt"), map_location=device)
        mymodel.load_state_dict(checkpoint['model'])

    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=1e-4,betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 5, min_lr = 1e-6)
    train_loader, validation_loader, test_loader = dataset.get_ARCADE_loaders(
        train_image_dir="./data/ARCADE/syntax/train/images/",
        train_mask_dir="./data/ARCADE/syntax/train/masks/",
        val_image_dir="./data/ARCADE/syntax/val/images/",
        val_mask_dir="./data/ARCADE/syntax/val/masks/",
        test_image_dir="./data/ARCADE/stenosis/test/images/",
        test_mask_dir="./data/ARCADE/stenosis/test/masks/",   
        batch_size=4)


    # train_loader, validation_loader, test_loader = dataset.get_XCAD_loaders(
    #     train_image_dir="./data/XCAD/train/images/",
    #     train_mask_dir="./data/XCAD/train/masks/",
    #     val_image_dir="./data/XCAD/val/images/",
    #     val_mask_dir="./data/XCAD/val/masks/",
    #     test_image_dir="./data/XCAD/test/images/",
    #     test_mask_dir="./data/XCAD/test/masks/",   
    #     batch_size=4)
    
    if trainable:
        history = {
            'epoch': [],
            'train_loss': [],
            'train_iou': [],
            'val_loss': [],
            'val_iou': []}
        EPOCHS = 400
        start_time = time.time()
        real_epoch = 0
        for epoch in range(EPOCHS):
            real_epoch = epoch
            train_loss, train_iou = train.model_train(dataloader=train_loader, model=mymodel, optimizer=optimizer,
                                                    device=device, config=config)
            val_loss, val_iou = train.model_evaluate(dataloader=validation_loader, model=mymodel, device=device, config=config)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_iou'].append(train_iou)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)

            save_path = os.path.join(path, f"history_no_frozen.csv")
            df = pd.DataFrame(history)
            df.to_csv(save_path, index=False)

            early_stopping(val_loss, mymodel, optimizer, scheduler)
            if early_stopping.early_stop:
                print(f"Early stopping triggered")
                break
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_epoch = total_time/real_epoch if real_epoch > 0 else 0
    if test: 
        # checkpoint = torch.load(os.path.join(path, 'checkpoint_fine_tune.pt'), map_location=device)
        # mymodel.load_state_dict(checkpoint['model'])
        test_loss, test_iou = train.model_evaluate(dataloader=test_loader, model=mymodel, device=device, config=config, mode='test')
        save_path = os.path.join(path, f"report_fine_tune.txt")
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(f"\nTest Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}\n")
            if trainable:
                f.write(f"\nTotal Time: {total_time:.1f} seconds, Average Time per Epoch: {avg_time_per_epoch:.1f} seconds\n")

# main_train(config_name='unet_baseline', test=True)
# main_train(config_name='unet_focal', test=True)
# main_train(config_name='unet_tversky', test=True)
# main_train(config_name='unet_focal_tversky', test=True)
# main_train(config_name='unet_combo', test=True)
# main_train(config_name='attention_unet_baseline', test=True)
# main_train(config_name='attention_unet_dice', test=True)
# main_train(config_name='unet_plus_baseline', test=True)
# main_train(config_name='unet_plus_dice', test=True)
# main_train(config_name='unet_dice_xcad', test=True)
# main_train(config_name='unet_dice_fine_tune', test=True, load='unet_baseline')
# main_train(config_name='attention_unet_dice_fine_tune', test=True, load='attention_unet_dice')
# main_train(config_name='unet_plus_dice_fine_tune', test=True, load='unet_plus_dice')
# main_train(config_name='unet_plus_enhanced', trainable=False, test=True, load='unet_plus_dice_fine_tune')
# main_train(config_name='unet_plus_enhanced_stenosis', trainable=False, test=True, load='unet_plus_dice_fine_tune')




import cv2
from skimage import measure
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import copy

def apply_postprocess_2d(mask_2d, close_ksize=7, erode_ksize=3):
    """2D 넘파이 배열에 대해 닫힘 -> 침식 -> 가장 큰 객체 추출을 수행합니다."""
    mask_uint8 = (mask_2d * 255).astype(np.uint8) if mask_2d.max() <= 1.0 else mask_2d.astype(np.uint8)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    eroded = cv2.erode(closed, kernel_erode, iterations=1)
    
    labels = measure.label(eroded > 0, connectivity=2)
    if labels.max() == 0:
        return (eroded > 0).astype(np.float32)
    
    bincount = np.bincount(labels.flat)
    bincount[0] = 0
    largest_label = bincount.argmax()
    return (labels == largest_label).astype(np.float32)

def extract_and_straighten(binary_mask):
    """마스크에서 중앙선을 추출하고 너비를 따라 일직선으로 폅니다."""
    mask_bool = binary_mask > 0
    dist_map = distance_transform_edt(mask_bool)
    skeleton = skeletonize(mask_bool)
    
    # 뼈대 픽셀 좌표 추출
    y_idx, x_idx = np.where(skeleton)
    points = list(zip(y_idx, x_idx))
    
    ordered_radii = []
    # 단순 그리디(Greedy) 방식으로 인접한 뼈대 픽셀을 순회하며 너비 추출
    if len(points) > 0:
        curr = points.pop(0)
        ordered_radii.append(dist_map[curr[0], curr[1]])
        while points:
            # 현재 점에서 가장 가까운 픽셀 찾기
            dists = [(p[0]-curr[0])**2 + (p[1]-curr[1])**2 for p in points]
            min_idx = np.argmin(dists)
            curr = points.pop(min_idx)
            ordered_radii.append(dist_map[curr[0], curr[1]])
            
    # Straightened 이미지 생성 (1차원 프로파일을 2D 튜브 형태로 시각화)
    length = len(ordered_radii)
    max_radius = int(np.ceil(max(ordered_radii))) if length > 0 else 10
    straight_img = np.zeros((max_radius * 2 + 20, length))
    
    center_y = max_radius + 10
    for x, r in enumerate(ordered_radii):
        straight_img[int(center_y - r) : int(center_y + r), x] = 1
        
    return skeleton, straight_img

# def visualize_single_prediction_pipeline(config_name, load_name):

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     with open('./segmentation_framework/config.yaml', 'r', encoding='utf-8') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)[config_name]
        
#     # 모델 로드
#     mymodel = model.select_model(config).to(device)
#     checkpoint_path = os.path.join(f"./segmentation_framework/{load_name}/checkpoint_fine_tune.pt")
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         mymodel.load_state_dict(checkpoint['model'])
#     mymodel.eval()

#     # 데이터 로더 (기존 설정과 동일하게 호출)
#     _, _, test_loader = dataset.get_ARCADE_loaders(
#         train_image_dir="./data/ARCADE/syntax/train/images/",
#         train_mask_dir="./data/ARCADE/syntax/train/masks/",
#         val_image_dir="./data/ARCADE/syntax/val/images/",
#         val_mask_dir="./data/ARCADE/syntax/val/masks/",
#         test_image_dir="./data/ARCADE/stenosis/test/images/",
#         test_mask_dir="./data/ARCADE/stenosis/test/masks/",   
#         batch_size=1  # 한 장만 시각화하기 위해 임시로 1로 설정
#     )

#     with torch.no_grad():
#         # 첫 번째 배치(이미지 1장) 가져오기
#         image, label = next(iter(test_loader))
#         image = image.to(device)
        
#         # 1. 원본 모델 추론 (Tiling)
#         tiles, _ = train.overlap_tiles(image, label)
#         tiles = tiles.repeat(1, 3, 1, 1)
#         out = mymodel(tiles)
#         if isinstance(out, tuple): out = out[0]
        
#         probs = torch.sigmoid(out).cpu().numpy()
#         merged_prob = train.merge_tiles(probs[0:4, 0, :, :])
#         original_pred_mask = (merged_prob > 0.5).astype(np.float32)

#         # 2. 후처리 (닫힘 + 미세 침식 + CCA)
#         post_processed_mask = apply_postprocess_2d(original_pred_mask, close_ksize=9, erode_ksize=3)

#         # 3 & 4. 중앙선 추출 및 Straightening
#         skeleton_mask, straight_img = extract_and_straighten(post_processed_mask)

#     # ---------------- 시각화 ----------------
#     fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
#     # (1) 원본 예측
#     axes[0].imshow(original_pred_mask, cmap='gray')
#     axes[0].set_title('1. Original Model Prediction', fontsize=14)
#     axes[0].axis('off')

#     # (2) 후처리 결과
#     axes[1].imshow(post_processed_mask, cmap='gray')
#     axes[1].set_title('2. Post-processed Mask', fontsize=14)
#     axes[1].axis('off')

#     # (3) 후처리 마스크 + 중앙선 오버레이
#     axes[2].imshow(post_processed_mask, cmap='gray')
#     axes[2].imshow(skeleton_mask, cmap='hot', alpha=0.5)
#     axes[2].set_title('3. Centerline on Post-processed', fontsize=14)
#     axes[2].axis('off')

#     # (4) Straightened 혈관 프로파일
#     axes[3].imshow(straight_img, cmap='gray')
#     axes[3].set_title('4. Straightened Vessel Profile', fontsize=14)
#     axes[3].axis('off')

#     plt.tight_layout()
#     plt.savefig(os.path.join(f"./segmentation_framework/{config_name}/prediction_visualization.png"))
#     plt.close(fig)
# === 스크립트 실행 ===
# 학습 코드가 끝난 후 이 함수가 호출되어 결과물이 시각화됩니다.
# visualize_single_prediction_pipeline(config_name='unet_plus_enhanced_stenosis_postprocessed', load_name='unet_plus_dice_fine_tune')

def save_all_postprocessed_masks_for_yolo(config_name, load_name, save_dir="./data/ARCADE/stenosis_yolo/unet_plus/images/train/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 환경 설정 및 모델 로드
    with open('./segmentation_framework/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[config_name]
        
    mymodel = model.select_model(config).to(device)
    checkpoint_path = os.path.join(f"./segmentation_framework/{load_name}/checkpoint_fine_tune.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        mymodel.load_state_dict(checkpoint['model'])
    mymodel.eval()

    # 2. 결과물을 저장할 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 3. 데이터 로더 호출 (전체 순회를 위해 사용)
    train_loader, val_loader, test_loader = dataset.get_ARCADE_loaders(
        train_image_dir="./data/ARCADE/stenosis/train/images/",
        train_mask_dir="./data/ARCADE/stenosis/train/masks/",
        val_image_dir="./data/ARCADE/stenosis/val/images/",
        val_mask_dir="./data/ARCADE/stenosis/val/masks/",
        test_image_dir="./data/ARCADE/stenosis/train/images/",
        test_mask_dir="./data/ARCADE/stenosis/train/masks/",   
        batch_size=1  # 한 장씩 독립적으로 후처리를 수행하기 위함
    )

    print(f"후처리가 완료된 마스크를 '{save_dir}' 경로에 저장합니다...")

    with torch.no_grad():
        # idx(Index)를 활용하여 순차적으로 접근
        for idx, batch_data in enumerate(test_loader):
            # 만약 DataLoader(데이터 로더)가 파일명을 반환한다면 분해해서 사용해야 합니다.
            # 예: image, label, filename = batch_data
            image, label = batch_data[0], batch_data[1]
            image = image.to(device)
            
            # 원본 추론 (Tiling 방식 유지)
            tiles, _ = train.overlap_tiles(image, label)
            tiles = tiles.repeat(1, 3, 1, 1)
            out = mymodel(tiles)
            if isinstance(out, tuple): 
                out = out[0]
            
            probs = torch.sigmoid(out).cpu().numpy()
            merged_prob = train.merge_tiles(probs[0:4, 0, :, :])
            original_pred_mask = (merged_prob > 0.5).astype(np.float32)

            # 후처리 (닫힘 -> 침식 -> LCC 추출)
            post_processed_mask = apply_postprocess_2d(original_pred_mask, close_ksize=9, erode_ksize=3)

            # 저장 포맷 변환: 모델 아웃풋(0.0 ~ 1.0)을 이미지 픽셀 값(0 ~ 255, uint8(Unsigned Integer 8-bit))으로 스케일링
            final_mask_uint8 = (post_processed_mask * 255).astype(np.uint8)

            # 파일명 지정 및 저장
            # 주의: 데이터 로더에서 원본 파일명을 반환받도록 수정하여 원본 이름과 동일하게 맞추는 것이
            # 나중에 YOLO(You Only Look Once) 라벨링과 매핑할 때 훨씬 안전합니다. 
            # 여기서는 파일명이 반환되지 않는다고 가정하고 임의의 인덱스를 부여했습니다.
            file_name = f"{idx+1}.png"
            save_path = os.path.join(save_dir, file_name)
            
            # OpenCV(Open Source Computer Vision Library)를 활용하여 512x512 PNG(Portable Network Graphics) 파일로 기록
            cv2.imwrite(save_path, final_mask_uint8)
            
            if (idx + 1) % 10 == 0:
                print(f"[{idx + 1}/{len(test_loader)}] 이미지 저장 완료...")

    print("모든 테스트 데이터의 전처리 및 저장이 성공적으로 완료되었습니다.")

# === 실행 ===
save_all_postprocessed_masks_for_yolo(config_name='unet_plus_enhanced_stenosis', load_name='unet_plus_dice_fine_tune')