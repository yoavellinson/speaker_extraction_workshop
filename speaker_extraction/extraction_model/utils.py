import torch
import torch
import numpy as np
import soundfile as sf

eps = torch.exp(torch.tensor(-6))

def padding_if_needed(x, y, ref=None):
    if ref == None:
        if x.shape[3] == y.shape[3]:
            return x, y
        elif x.shape[3] > y.shape[3]:
            y_padded = torch.zeros(
                (y.shape[0], y.shape[1], y.shape[2], y.shape[3]+1), dtype=y.dtype, device=y.device)
            y_padded[:, :, :, :y.shape[3]] = y
            return x, y_padded
        elif x.shape[3] < y.shape[3]:
            x_padded = torch.zeros(
                (x.shape[0], x.shape[1], x.shape[2], x.shape[3]+1), dtype=x.dtype, device=x.device)
            x_padded[:, :, :, :x.shape[3]] = x
            return x_padded, y

    else:
        
        if x.shape[3] > y.shape[3]:
            y_padded = torch.zeros(
                (y.shape[0], y.shape[1], y.shape[2], y.shape[3]+1), dtype=y.dtype, device=y.device)
            y_padded[:, :, :, :y.shape[3]] = y

            ref_padded = torch.zeros(
                (ref.shape[0], ref.shape[1], ref.shape[2], ref.shape[3]+1), dtype=ref.dtype, device=ref.device)
            ref_padded[:, :, :, :y.shape[3]] = ref
            return x, y_padded, ref_padded
        elif x.shape[3] < y.shape[3]:
            x_padded = torch.zeros(
                (x.shape[0], x.shape[1], x.shape[2], x.shape[3]+1), dtype=x.dtype, device=x.device)
            x_padded[:, :, :, :x.shape[3]] = x
            return x_padded, y, ref
        else:
            return x, y, ref

def save_wave(y, path, fs=8000):
    if not type(y).__module__ == np.__name__:
        y = np.squeeze(y.cpu().detach().numpy())
    y = y / 1.1 / np.max(np.abs(y))
    sf.write(path, y,fs)

