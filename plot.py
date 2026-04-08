import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dataset
import model
import train
import os
import yaml

# 1. 데이터 로드 및 모델 입력용 전처리
image_tensor, mask_tensor = dataset.ARCADE_eval_dataset('./data/XCAD/test/images/', 
                                                 './data/XCAD/test/masks/')[1]
image_tensor = image_tensor.repeat(3, 1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델에 넣기 위해 Batch(배치) 차원 추가: [C, H, W] -> [1, C, H, W]
input_tensor = image_tensor.unsqueeze(0).to(device) 

with open(f'./segmentation_framework/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
att_config = config['attention_unet_dice_fine_tune']

# 2. 모델 로드 및 평가 모드 설정 (가중치가 로드되어 있다고 가정)
att_model = model.select_model(att_config).to(device)
plus_config = config['unet_plus_dice_fine_tune']
plus_model = model.select_model(plus_config).to(device)

checkpoint = torch.load(os.path.join(f"./segmentation_framework/attention_unet_dice_fine_tune/checkpoint_fine_tune.pt"), map_location=device)
att_model.load_state_dict(checkpoint['model'])
checkpoint2 = torch.load(os.path.join(f"./segmentation_framework/unet_plus_dice_fine_tune/checkpoint_fine_tune.pt"), map_location=device)
plus_model.load_state_dict(checkpoint2['model'])

att_model.eval()
plus_model.eval()

# 3. 추론 진행 (Attention Map인 psi2, psi3, psi4 반환 추가 반영)
with torch.no_grad():
    att_final, att_out1, att_out2, att_out3, psi2, psi3, psi4 = att_model(input_tensor)
    plus_final, plus_out3, plus_out2, plus_out1 = plus_model(input_tensor)

# 4. 텐서 후처리 (순서: Deep -> Shallow -> Final)
# Attention U-Net 출력물 (Deep: out3 -> Shallow: out1 -> Final)
att_vis = [
    torch.sigmoid(att_out3).squeeze().cpu().numpy(),
    torch.sigmoid(att_out2).squeeze().cpu().numpy(),
    torch.sigmoid(att_out1).squeeze().cpu().numpy(),
    torch.sigmoid(att_final).squeeze().cpu().numpy()
]

# Attention Map (모델 내부에서 이미 Sigmoid를 거쳤으므로 그대로 사용)
# 순서: Deep: psi4 -> Shallow: psi2
att_maps = [
    psi4.squeeze().cpu().numpy(),
    psi3.squeeze().cpu().numpy(),
    psi2.squeeze().cpu().numpy(),
    None # Final 단계는 Attention Map이 없으므로 비워둠
]

# U-Net++ 출력물 (Deep: out3 -> Shallow: out1 -> Final)
plus_vis = [
    torch.sigmoid(plus_out3).squeeze().cpu().numpy(),
    torch.sigmoid(plus_out2).squeeze().cpu().numpy(),
    torch.sigmoid(plus_out1).squeeze().cpu().numpy(),
    torch.sigmoid(plus_final).squeeze().cpu().numpy()
]

# 5. 시각화 및 저장 (3행 4열 구조)
fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=300)

# 첫 번째 행: Attention U-Net 결과물
titles_att = ['Att-UNet: Out3 (Deep)', 'Att-UNet: Out2', 'Att-UNet: Out1 (Shallow)', 'Att-UNet: Final']
for i in range(4):
    axes[0, i].imshow(att_vis[i], cmap='gray')
    axes[0, i].set_title(titles_att[i])

# 두 번째 행: Attention Maps
titles_maps = ['Att Map: Psi4 (Deep)', 'Att Map: Psi3', 'Att Map: Psi2 (Shallow)', '']
for i in range(3):
    # Attention Map은 값의 분포(0~1)를 더 명확히 보기 위해 'gray' 또는 열화상 느낌의 'viridis' 등을 쓸 수 있습니다. 
    # 여기서는 직관적인 형태 확인을 위해 gray를 유지합니다.
    axes[1, i].imshow(att_maps[i], cmap='gray')
    axes[1, i].set_title(titles_maps[i])
axes[1, 3].axis('off') # 4번째 열은 데이터가 없으므로 축(Axis) 렌더링 끄기

# 세 번째 행: U-Net++ 결과물
titles_plus = ['UNet++: out_x0_3 (Deep)', 'UNet++: out_x0_2', 'UNet++: out_x0_1 (Shallow)', 'UNet++: Final']
for i in range(4):
    axes[2, i].imshow(plus_vis[i], cmap='gray')
    axes[2, i].set_title(titles_plus[i])

plt.tight_layout()
plt.savefig('./segmentation_framework/supervision_model_outputs.png', bbox_inches='tight')
plt.close(fig)