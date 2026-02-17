import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, out_dir, filename='last.pth'):
    import os
    torch.save(state, os.path.join(out_dir, filename))
    if is_best:
        torch.save(state, os.path.join(out_dir, 'best.pth'))

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint
