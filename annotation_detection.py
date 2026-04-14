import cv2
import os
import random
import glob

def crop_patches_yolo(img_dir, label_dir, out_img_dir, out_label_dir, patch_size=640):
    """
    이미지와 YOLO(You Only Look Once) 라벨을 불러와 타겟 포함/미포함 1:1 비율로 패치를 생성합니다.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # 이미지 파일 목록 불러오기 (jpg, png 등)
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))

    for img_path in img_paths:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        label_path = os.path.join(label_dir, name + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w, _ = img.shape
        boxes = []

        # 1. 라벨 파일 읽어오기 및 픽셀 좌표계로 변환
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        # 정규화된 좌표를 원본 픽셀 좌표로 변환
                        xmin = int((cx - bw / 2) * w)
                        ymin = int((cy - bh / 2) * h)
                        xmax = int((cx + bw / 2) * w)
                        ymax = int((cy + bh / 2) * h)
                        boxes.append([class_id, xmin, ymin, xmax, ymax])

        pos_patch_count = 0

        # 2. 타겟 포함(Positive) 패치 생성
        for idx, box in enumerate(boxes):
            c_id, xmin, ymin, xmax, ymax = box

            # 박스가 패치 사이즈보다 크면 예외 처리 (관상동맥 협착은 보통 패치보다 작음)
            if (xmax - xmin) > patch_size or (ymax - ymin) > patch_size:
                continue

            # 박스가 패치 안에 무조건 포함되도록 크롭 시작점(x, y) 범위 설정
            px_min = max(0, xmax - patch_size)
            px_max = min(w - patch_size, xmin)
            py_min = max(0, ymax - patch_size)
            py_max = min(h - patch_size, ymin)

            if px_min > px_max or py_min > py_max:
                continue

            # 랜덤한 위치로 크롭 (Data Augmentation 효과)
            px = random.randint(px_min, px_max)
            py = random.randint(py_min, py_max)

            patch_img = img[py:py+patch_size, px:px+patch_size]
            patch_boxes = []

            # 잘라낸 패치 안에 들어오는 모든 타겟의 좌표를 패치 기준으로 재계산
            for b in boxes:
                bc_id, bxmin, bymin, bxmax, bymax = b
                
                # 교집합 영역 계산
                inter_xmin = max(px, bxmin)
                inter_ymin = max(py, bymin)
                inter_xmax = min(px + patch_size, bxmax)
                inter_ymax = min(py + patch_size, bymax)

                # 패치 안에 객체의 일부라도 포함되어 있다면
                if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
                    # 패치 기준 픽셀 좌표로 변환
                    new_xmin = inter_xmin - px
                    new_ymin = inter_ymin - py
                    new_xmax = inter_xmax - px
                    new_ymax = inter_ymax - py

                    # YOLO(You Only Look Once) 정규화 좌표(0~1)로 다시 변환
                    new_cx = (new_xmin + new_xmax) / 2.0 / patch_size
                    new_cy = (new_ymin + new_ymax) / 2.0 / patch_size
                    new_bw = (new_xmax - new_xmin) / patch_size
                    new_bh = (new_ymax - new_ymin) / patch_size

                    patch_boxes.append(f"{bc_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

            if patch_boxes:
                out_name = f"{name}_pos_{idx}"
                cv2.imwrite(os.path.join(out_img_dir, f"{out_name}{ext}"), patch_img)
                with open(os.path.join(out_label_dir, f"{out_name}.txt"), 'w') as f:
                    f.write('\n'.join(patch_boxes))
                pos_patch_count += 1

        # 3. 타겟 미포함(Negative) 패치 생성 (1:1 비율 유지)
        neg_generated = 0
        attempts = 0  # 무한 루프 방지용
        
        while neg_generated < pos_patch_count and attempts < 100:
            attempts += 1
            px = random.randint(0, w - patch_size)
            py = random.randint(0, h - patch_size)

            has_target = False
            # 잘라낼 영역에 타겟이 하나라도 걸치는지 검사
            for b in boxes:
                _, bxmin, bymin, bxmax, bymax = b
                inter_xmin = max(px, bxmin)
                inter_ymin = max(py, bymin)
                inter_xmax = min(px + patch_size, bxmax)
                inter_ymax = min(py + patch_size, bymax)

                if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
                    has_target = True
                    break

            # 타겟이 전혀 없는 순수 배경 패치일 경우 저장
            if not has_target:
                patch_img = img[py:py+patch_size, px:px+patch_size]
                out_name = f"{name}_neg_{neg_generated}"
                cv2.imwrite(os.path.join(out_img_dir, f"{out_name}{ext}"), patch_img)
                
                # 빈 txt 파일 생성 (YOLO는 빈 파일 = Background 처리)
                open(os.path.join(out_label_dir, f"{out_name}.txt"), 'w').close()
                neg_generated += 1

    print("패치 생성 및 라벨링 재계산이 완료되었습니다.")

# 사용 예시 (본인의 디렉토리 경로에 맞게 수정하여 사용하십시오)
if __name__ == "__main__":
    # 원본 데이터 경로
    input_images = "./data/ARCADE/stenosis_yolo/images/test"
    input_labels = "./data/ARCADE/stenosis_yolo/labels/test"
    
    # 패치 저장 경로
    output_images = "./data/ARCADE/stenosis_yolo/patches/images/test"
    output_labels = "./data/ARCADE/stenosis_yolo/patches/labels/test"
    
    # 함수 실행 (원하는 패치 사이즈 입력, 기본 640)
    crop_patches_yolo(input_images, input_labels, output_images, output_labels, patch_size=384)