import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data import ConcreteClassificationDataset, get_classification_filepaths, load_split_indices, ConcreteSegmentationDataset, get_segmentation_filepaths
from src.models import ClassificationModel, UNet
from src.metrics import classification_metrics, dice_coef, iou_coef
from src.train_utils import set_seed
from src.viz import show_batch, overlay_mask


def evaluate_classification(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((probs > 0.5).astype(int))
    metrics = classification_metrics(y_true, y_pred, y_prob)
    print("Classification metrics:", metrics)
    return metrics

def evaluate_segmentation(model, loader, device):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            dices.append(dice_coef(logits, masks))
            ious.append(iou_coef(logits, masks))
    print(f"Segmentation Dice: {sum(dices)/len(dices):.4f}, IoU: {sum(ious)/len(ious):.4f}")
    return dices, ious

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.task == 'classification':
        filepaths, labels = get_classification_filepaths(args.data_root)
        splits = load_split_indices(os.path.join('runs/classification', 'splits.json'))
        split_files = splits[args.split]
        split_labels = [labels[filepaths.index(f)] for f in split_files]
        val_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ds = ConcreteClassificationDataset(split_files, split_labels, transform=val_transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        model = ClassificationModel()
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
        model = model.to(device)
        evaluate_classification(model, loader, device)
    elif args.task == 'segmentation':
        if not os.path.exists(os.path.join(args.data_root, 'segmentation')):
            print('Segmentation data not found. Exiting.')
            return
        img_paths, mask_paths = get_segmentation_filepaths(args.data_root)
        val_transform = None  # Should match training val_transform
        ds = ConcreteSegmentationDataset(img_paths, mask_paths, transform=val_transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        model = UNet()
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
        model = model.to(device)
        evaluate_segmentation(model, loader, device)

if __name__ == '__main__':
    main()
