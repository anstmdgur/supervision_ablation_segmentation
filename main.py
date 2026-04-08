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

def main_train(config_name, test = True, load = False):
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
        checkpoint = torch.load(os.path.join(f"./segmentation_framework/{load}/checkpoint_no_frozen.pt"), map_location=device)
        mymodel.load_state_dict(checkpoint['model'])

    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=1e-4,betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 5, min_lr = 1e-6)
    # train_loader, validation_loader, test_loader = dataset.get_ARCADE_loaders(
    #     train_image_dir="./data/ARCADE/syntax/train/images/",
    #     train_mask_dir="./data/ARCADE/syntax/train/masks/",
    #     val_image_dir="./data/ARCADE/syntax/val/images/",
    #     val_mask_dir="./data/ARCADE/syntax/val/masks/",
    #     test_image_dir="./data/ARCADE/syntax/test/images/",
    #     test_mask_dir="./data/ARCADE/syntax/test/masks/",   
    #     batch_size=4)


    train_loader, validation_loader, test_loader = dataset.get_XCAD_loaders(
        train_image_dir="./data/XCAD/train/images/",
        train_mask_dir="./data/XCAD/train/masks/",
        val_image_dir="./data/XCAD/val/images/",
        val_mask_dir="./data/XCAD/val/masks/",
        test_image_dir="./data/XCAD/test/images/",
        test_mask_dir="./data/XCAD/test/masks/",   
        batch_size=4)
    

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
        checkpoint = torch.load(os.path.join(path, 'checkpoint_fine_tune.pt'), map_location=device)
        mymodel.load_state_dict(checkpoint['model'])
        test_loss, test_iou = train.model_evaluate(dataloader=test_loader, model=mymodel, device=device, config=config, mode='test')
        save_path = os.path.join(path, f"report_fine_tune.txt")
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(f"\nTest Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}\n")
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
main_train(config_name='attention_unet_dice_fine_tune', test=True, load='attention_unet_dice')
main_train(config_name='unet_plus_dice_fine_tune', test=True, load='unet_plus_dice')