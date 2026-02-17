import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from albumentations import Compose, HorizontalFlip, RandomRotate90, RandomCrop, Normalize
from albumentations.pytorch import ToTensorV2
from src.data import get_segmentation_filepaths, ConcreteSegmentationDataset
from src.models import UNet
from src.losses import get_loss
from src.metrics import dice_coef, iou_coef
from src.train_utils import set_seed, save_checkpoint
from src.viz import plot_learning_curves, overlay_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='runs/segmentation')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='bce_dice')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(os.path.join(args.data_root, 'segmentation')):
        print('Segmentation data not found. Exiting.')
        return

    img_paths, mask_paths = get_segmentation_filepaths(args.data_root)
    n = len(img_paths)
    idxs = list(range(n))
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(args.seed)
    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(idxs, test_size=0.2, random_state=args.seed)
    train_imgs = [img_paths[i] for i in idx_train]
    train_masks = [mask_paths[i] for i in idx_train]
    val_imgs = [img_paths[i] for i in idx_val]
    val_masks = [mask_paths[i] for i in idx_val]

    train_transform = Compose([
        HorizontalFlip(),
        RandomRotate90(),
        RandomCrop(227, 227),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_ds = ConcreteSegmentationDataset(train_imgs, train_masks, transform=train_transform)
    val_ds = ConcreteSegmentationDataset(val_imgs, val_masks, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = UNet(activation=args.activation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = get_loss(args.loss)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0
    patience = args.patience
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                dices.append(dice_coef(logits, masks))
        val_loss /= len(val_loader.dataset)
        val_dice = np.mean(dices)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")
        save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, val_dice > best_dice, args.out_dir)
        if val_dice > best_dice:
            best_dice = val_dice
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break
        scheduler.step()
    plot_learning_curves(history, save_path=os.path.join(args.out_dir, 'learning_curves.png'))
    print("Training complete. Best val Dice:", best_dice)

if __name__ == '__main__':
    main()
