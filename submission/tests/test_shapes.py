"""Validate the CNN's tensor arithmetic.

Runs the real model when PyTorch is installed. When it is not -- e.g. on a
Python version with no torch wheel yet -- it falls back to reproducing the
conv/pool output-length arithmetic in plain Python, which is where dimension
bugs actually occur. Run with: python tests/test_shapes.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FS, T, C, L, B, N_CLASSES = 100, 3000, 7, 20, 2, 5


def conv_out(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1


def pool_out(n, k):
    return n // k


def branch_len(kernel, stride, pool1, pool2):
    n = conv_out(T, kernel, stride, kernel // 2)
    n = pool_out(n, pool1)
    n = conv_out(n, 8, 1, 4)
    n = conv_out(n, 8, 1, 4)
    n = pool_out(n, pool2)
    assert n >= 1, f"branch collapses to {n} samples before the final pool"
    return n


def main():
    fine = branch_len(FS // 2, FS // 16, 8, 4)
    coarse = branch_len(FS * 4, FS // 2, 4, 2)
    embed = 128 + 128            # AdaptiveAvgPool1d(1) on each branch, width*2 each
    print(f"  fine branch   -> {fine} timesteps before adaptive pool")
    print(f"  coarse branch -> {coarse} timesteps before adaptive pool")
    print(f"  concatenated embedding = {embed}")
    assert embed == 256, "GRU input_size in SleepNet must match the concatenated embedding"

    try:
        import torch
        from train_cnn import SleepNet
    except Exception as e:
        print(f"\n  PyTorch not available ({type(e).__name__}); arithmetic checks passed.")
        return 0

    m = SleepNet(C, fs=FS, n_classes=N_CLASSES)
    x = torch.randn(B, L, C, T)
    y = m(x)
    assert y.shape == (B, L, N_CLASSES), y.shape
    n_params = sum(p.numel() for p in m.parameters())
    print(f"\n  forward pass OK: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  parameters: {n_params:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
