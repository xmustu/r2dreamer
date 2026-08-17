"""Custom HDF5 dataset loader for LeWorldModel training.

Key fix: pre-loads metadata and small columns into memory; opens HDF5
per-request for pixel data to avoid fork-safe file handle issues.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import threading


class HDF5Dataset(Dataset):
    """Fork-safe HDF5 dataset for LeWorldModel.

    Pre-loads action, state, proprio, episode_idx, step_idx into memory.
    Reads pixels from HDF5 on each __getitem__ (with a thread-local file handle
    for efficiency when num_workers=0).
    """

    def __init__(self, path: str, keys_to_load=None, keys_to_cache=None,
                 num_steps: int = 4, frameskip: int = 1):
        self.path = path
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.transform = None

        # Load everything into memory (60K × 224 × 224 × 3 ≈ 9GB — fits in system RAM)
        with h5py.File(path, "r") as f:
            self._n = len(f["pixels"])
            print(f"[HDF5] Loading {self._n} frames into memory ({f['pixels'].shape} pixels)...")
            self._pixels = f["pixels"][:]  # Pre-load all pixels!
            self._ep_idx = f["episode_idx"][:]
            self._step_idx = f["step_idx"][:]
            self._action = f["action"][:]
            self._state = f["state"][:]
            self._proprio = f["proprio"][:]
        print(f"[HDF5] Loaded. Pixels: {self._pixels.nbytes/1024**3:.1f} GB, "
              f"Total: {(self._pixels.nbytes + self._action.nbytes + self._state.nbytes)/1024**3:.1f} GB")

        # Build valid starting indices
        self._valid_starts = []
        for i in range(self._n - num_steps * frameskip):
            if self._ep_idx[i] == self._ep_idx[i + num_steps * frameskip - 1]:
                ok = True
                for j in range(1, num_steps):
                    expected = self._step_idx[i] + j * frameskip
                    if self._step_idx[i + j * frameskip] != expected:
                        ok = False
                        break
                if ok:
                    self._valid_starts.append(i)
        self._valid_starts = np.array(self._valid_starts)

    @property
    def column_names(self):
        return ["pixels", "action", "state", "proprio", "episode_idx", "step_idx"]

    def __len__(self):
        return len(self._valid_starts)

    def get_col_data(self, col: str):
        """Return all data for a column (for normalization)."""
        mapping = {
            "action": self._action, "state": self._state,
            "proprio": self._proprio, "episode_idx": self._ep_idx,
            "step_idx": self._step_idx,
        }
        if col in mapping:
            return mapping[col]
        if col == "pixels":
            with h5py.File(self.path, "r") as f:
                return f["pixels"][:]
        raise KeyError(col)

    def get_dim(self, col: str):
        if col == "pixels":
            return 0
        dims = {"action": self._action.shape[-1], "state": self._state.shape[-1],
                "proprio": self._proprio.shape[-1]}
        return dims.get(col, 0)

    def get_row_data(self, indices):
        """Return dict with data for specific indices (int or list)."""
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        indices = np.array(indices)
        return {
            "pixels": self._read_pixels(indices),
            "action": self._action[indices],
            "state": self._state[indices],
            "proprio": self._proprio[indices],
            "episode_idx": self._ep_idx[indices],
            "step_idx": self._step_idx[indices],
        }

    def _read_pixels(self, indices):
        """Read pixel data for given indices from HDF5."""
        with h5py.File(self.path, "r") as f:
            return f["pixels"][indices]

    def __getitem__(self, idx):
        start = self._valid_starts[idx]
        indices = [start + i * self.frameskip for i in range(self.num_steps)]

        # Read pixel data
        pixels_raw = self._read_pixels(indices)  # (T, H, W, C) uint8

        item = {}
        # pixels: uint8 -> float32 [0,1] -> (T, C, H, W)
        arr = pixels_raw.astype(np.float32) / 255.0
        arr = arr.transpose(0, 3, 1, 2)
        item["pixels"] = torch.from_numpy(arr)

        item["action"] = torch.from_numpy(self._action[indices].astype(np.float32))
        item["state"] = torch.from_numpy(self._state[indices].astype(np.float32))
        item["proprio"] = torch.from_numpy(self._proprio[indices].astype(np.float32))
        item["episode_idx"] = torch.from_numpy(self._ep_idx[indices].astype(np.float32))
        item["step_idx"] = torch.from_numpy(self._step_idx[indices].astype(np.float32))

        if self.transform is not None:
            item = self.transform(item)

        return item


def load_hdf5_dataset(path: str, num_steps: int = 4, frameskip: int = 5, **kwargs):
    return HDF5Dataset(path, num_steps=num_steps, frameskip=frameskip)
