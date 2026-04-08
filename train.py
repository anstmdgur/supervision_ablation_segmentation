import torch
import torch.nn as nn
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torchmetrics as tm
import cv2
import os
import seaborn as sns


def overlap_tiles(images, masks, tile_size=384, stride=128):

    B, C, H, W = images.shape #4,1(3),512,512 등
    tile_images = []
    tile_masks = []
    

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile_images.append(images[:, :, y:y + tile_size, x:x + tile_size])
            tile_masks.append(masks[:, :, y:y + tile_size, x:x + tile_size])
            
    img_stacked = torch.stack(tile_images, dim=1) 
    mask_stacked = torch.stack(tile_masks, dim=1) # 결과 차원: [B, 4, C, 384, 384]
    
    # B와 4 차원을 하나로 합쳐서 [B*4, C, 384, 384]로 평탄화
    # 이제 순서는 [이미지1_타일1, 이미지1_타일2, 이미지1_타일3, 이미지1_타일4, 이미지2_타일1...] 이 됨
    img_out = img_stacked.reshape(-1, C, tile_size, tile_size)
    mask_out = mask_stacked.reshape(-1, C, tile_size, tile_size)
    return img_out, mask_out

def get_gaussian_window(tile_size=384, sigma=96): #표준편차96. 384의 1/4

    kernel_1d = cv2.getGaussianKernel(tile_size, sigma) #1차원 가우시안 커널

    window_2d = kernel_1d @ kernel_1d.T #2차원 가우시안 윈도우 생성. @는 행렬 곱셈 연산자. kernel_1d의 전치행렬과 곱하여 2D 가우시안 윈도우를 만듦
    #384*1 @ 1*384 -> 384*384
    window_2d = window_2d / window_2d.max() #윈도우의 최대값으로 나누어 정규화하여 0과 1 사이의 값이 되도록 함.
    #getGaussianKernel은 다 더했을때 1이 나오도록 설계된 블러 필터용 함수이기 때문에 최대값(예를들면 0.04..)으로 나눠서 0~1(중앙 가장 큰값이 1)로 만듦
    return window_2d


def merge_tiles(predicted_tiles, h=512, w=512, tile_size=384, stride=128):
    canvas = np.zeros((h, w))       #최종 예측용 도화지
    trust_sum = np.zeros((h, w))    #신뢰도 맵
    
    trust_map = get_gaussian_window()
    
    tile_idx = 0

    for y in range(0, h - tile_size + 1, stride): #0,128,128
        for x in range(0, w - tile_size + 1, stride):
            
            weighted_pred = predicted_tiles[tile_idx] * trust_map # *요소곱으로 weighted prediction 구함
            
            canvas[y:y + tile_size, x:x + tile_size] += weighted_pred #신뢰도가 곱해진 예측값 더해줌
            
            trust_sum[y:y + tile_size, x:x + tile_size] += trust_map #신뢰도도 따로 더해서 저장해둠
            
            tile_idx += 1

    final_prediction = canvas / (trust_sum + 1e-8)
    
    return final_prediction



dice_loss = smp.losses.DiceLoss(mode='binary')
focal_loss = smp.losses.FocalLoss(mode='binary',alpha=0.5,gamma=2.0)
tversky_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7)
focal_tversky_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, gamma=2.0)
def combo_loss(outputs, masks, alpha=0.5):
    dice = dice_loss(outputs, masks)
    focal = focal_loss(outputs, masks)
    return alpha * dice + (1 - alpha) * focal

loss_functions = {
        'dice': dice_loss,
        'focal': focal_loss,
        'tversky': tversky_loss,
        'focal_tversky': focal_tversky_loss,
        'combo': combo_loss
}


def model_train(dataloader, model, optimizer, device, config):
    
    model.train()
    iou_sum = 0.0
    train_loss_sum = 0.0
    loss_type = config.get('loss_type')
    iou_score = tm.classification.BinaryJaccardIndex().to(device)
    loss_function = loss_functions[loss_type]

    ds_weights = [1.0, 0.5, 0.25, 0.125] #깊은 레이어의 출력일수록 가중치를 낮게

    for image, label in dataloader:
        images = image.to(device)
        labels = label.to(device)
        tiles, masks = overlap_tiles(images, labels) #gpu에서 타일링 수행
        tiles = tiles.repeat(1, 3, 1, 1)
        outputs = model(tiles)
        if isinstance(outputs, tuple): #출력이 튜플 (supervision이 여러개)
            loss = 0
            for i, out in enumerate(outputs):
                # attention unet의 경우 마스크보다 out의 해상도가 작을 수 있음.
                if out.shape[2:] != masks.shape[2:]:
                    out = F.interpolate(out, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                weight = ds_weights[i] if i < len(ds_weights) else 0.1
                loss += weight * loss_function(out, masks)
            
            # IoU(Intersection over Union) 계산은 가장 성능이 좋은 최종 출력(outputs[0])으로만 수행
            main_output = outputs[0] 
        else:
            loss = loss_function(outputs, masks)
            main_output = outputs
            
        train_loss_sum += loss.item()
        
        iou = iou_score(main_output, masks)
        iou_sum += iou.item()
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

    avg_loss = train_loss_sum / len(dataloader)
    avg_iou = iou_sum / len(dataloader)

    return avg_loss, avg_iou

def model_evaluate(dataloader, model, device, config, mode = 'validation'):
    model.eval()
    iou_sum = 0
    eval_loss_sum = 0
    loss_type = config.get('loss_type')
    path = config.get('path')
    iou_score = tm.classification.BinaryJaccardIndex().to(device)
    loss_function = loss_functions[loss_type]
    if mode == 'test':
        test_predictions = []
        test_labels = []
        test_images = []
    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)
            tiles, masks = overlap_tiles(image, label)
            tiles = tiles.repeat(1, 3, 1, 1)
            output_tiles = model(tiles)

            if isinstance(output_tiles, tuple):
                main_output_tiles = output_tiles[0]
            else:
                main_output_tiles = output_tiles
            
            loss = loss_function(main_output_tiles, masks)
            eval_loss_sum += loss.item()


            probs_tiles_np = torch.sigmoid(main_output_tiles).cpu().numpy()
            merged_probs = []

            for b in range(image.size(0)): #batch size만큼 반복
                item_tiles = probs_tiles_np[b*4 : (b+1)*4, 0, :, :] # 4개 타일 추출
                merged = merge_tiles(item_tiles)
                merged_probs.append(merged)

            merged_probs_tensor = torch.tensor(np.array(merged_probs)).unsqueeze(1).to(device) # (B, 1, H, W) 텐서로 만듦

            iou = iou_score(merged_probs_tensor, label)
            iou_sum += iou.item()
            if mode == 'test':
                output = (merged_probs_tensor > 0.5).float()
                
                test_predictions.extend(output.cpu().numpy())
                test_labels.extend(label.cpu().numpy())
                test_images.extend(image.cpu().numpy())

    if mode == 'test':
        i=1
        for pred, label, image in zip(test_predictions, test_labels, test_images):
            pred = pred.squeeze()
            label = label.squeeze()
            image = image[0] # 원본 이미지 (채널 제거)

            mean = 0.449
            std = 0.226
            image = (image * std) + mean
            image = np.clip(image, 0.0, 1.0)#floating point error로 인한 0~1 범위 벗어나는거 방지

            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi = 200)
            axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f'Image')

            axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Prediction')

            axes[2].imshow(label, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f'Ground Truth')
            save_path = os.path.join(path, f"result/{i}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            i+=1

        test_labels = np.array(test_labels).flatten()
        test_predictions = np.array(test_predictions).flatten()
        report = classification_report(test_labels, test_predictions, digits=4, target_names=['background', 'foreground'])
        save_path = os.path.join(path, f"report_fine_tune.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

        save_path = os.path.join(path, f"confusion_matrix_fine_tune.png")
        cm = confusion_matrix(test_labels,test_predictions)
        fig = plt.figure(figsize=(10,10),dpi= 100)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['background', 'foreground'], yticklabels=['background', 'foreground'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    avg_loss = eval_loss_sum / len(dataloader)
    avg_iou = iou_sum / len(dataloader)

    return avg_loss, avg_iou


class EarlyStopping():
    def __init__(self,path,patience = 10, delta = 0.0005,verbose = True):
        self.patience = patience 
        self.delta = delta 
        self.verbose = verbose
        self.path = path

        self.count = 0
        self.best_score = None
        self.early_stop = False 
        self.val_loss_min = float('inf')

    def save_checkpoint(self, val_loss, model,optimizer,scheduler):
        path = os.path.join(self.path, 'checkpoint_fine_tune.pt')  #ex) f'./segmentation_framework/unet_baseline/checkpoint.pt'
        if self.verbose :
            print(f"Validation loss decreased to {val_loss}. save model\n")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, path)
            self.val_loss_min = val_loss

    def __call__(self, val_loss, model, optimizer, scheduler):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,scheduler)
            return 1
        elif score < self.best_score + self.delta: 
            self.count += 1
            if self.verbose:
                print(f"EarlyStopping count: {self.count}\n")
            if self.count >= self.patience:
                self.early_stop = True
                return 0
        else: 
            self.best_score = score
            self.save_checkpoint(val_loss,model,optimizer,scheduler)
            self.count = 0
            return 1