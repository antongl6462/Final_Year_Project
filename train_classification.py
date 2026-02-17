import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data import get_classification_filepaths, stratified_split, ConcreteClassificationDataset, save_split_indices
from src.models import ClassificationModel
from src.losses import get_loss
from src.metrics import classification_metrics, threshold_predictions
from src.train_utils import set_seed, save_checkpoint
from src.viz import plot_learning_curves, show_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='runs/classification')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--cutmix', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    filepaths, labels = get_classification_filepaths(args.data_root)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(filepaths, labels, seed=args.seed)
    save_split_indices({'train': X_train, 'val': X_val, 'test': X_test}, os.path.join(args.out_dir, 'splits.json'))

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(227, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ConcreteClassificationDataset(X_train, y_train, transform=train_transform)
    val_ds = ConcreteClassificationDataset(X_val, y_val, transform=val_transform)
    test_ds = ConcreteClassificationDataset(X_test, y_test, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ClassificationModel(backbone=args.backbone, dropout=args.dropout, activation=args.activation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = get_loss('bce')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    patience = args.patience
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.float().to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob.extend(probs)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend((probs > 0.5).astype(int))
        val_loss /= len(val_loader.dataset)
        metrics = classification_metrics(y_true, y_pred, y_prob)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(metrics['f1'])
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={metrics['f1']:.4f}")
        save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, metrics['f1'] > best_f1, args.out_dir)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break
        scheduler.step()
    plot_learning_curves(history, save_path=os.path.join(args.out_dir, 'learning_curves.png'))
    print("Training complete. Best val F1:", best_f1)

if __name__ == '__main__':
    main()
