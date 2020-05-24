import numpy as np
import torch

def collate_fn(batch):
    # mainly used to convert numpy to tensor and concatenate
    samples = list(d for d in batch if d is not None)
    if len(samples) == 0:
        print("current batch is empty")
        return None

    
    
