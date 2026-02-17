import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_learning_curves(history, save_path=None):
    plt.figure(figsize=(10,5))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Learning Curves')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def show_batch(images, labels=None, preds=None, n=8):
    images = images[:n]
    plt.figure(figsize=(n*2, 2))
    for i, img in enumerate(images):
        plt.subplot(1, n, i+1)
        img = img.permute(1,2,0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        plt.imshow(img)
        title = ''
        if labels is not None:
            title += f'T:{labels[i]}'
        if preds is not None:
            title += f'\nP:{preds[i]}'
        plt.title(title)
        plt.axis('off')
    plt.show()

def overlay_mask(image, mask, alpha=0.5):
    if isinstance(image, torch.Tensor):
        image = image.permute(1,2,0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-5)
    mask = (mask > 0.5).astype(np.float32)
    overlay = image.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask * alpha, 0, 1)
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
