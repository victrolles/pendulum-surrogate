import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PendulumDataset(Dataset):
    """Read compact trajectories or legacy rows as single-step training pairs."""

    def __init__(self, path: str | Path):
        self._rows = None
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(archive["metadata"].item()) if "metadata" in archive.files else {}

            if "dataset" in archive.files:
                rows = archive["dataset"]
                if rows.ndim != 2 or rows.shape[1] != 7:
                    raise ValueError("A legacy dataset must have seven transition columns.")
                self._rows = torch.from_numpy(rows.astype(np.float32, copy=False))
                return

            if not {"parameters", "states"}.issubset(archive.files):
                raise ValueError("A compact dataset must contain parameters and states.")
            parameters = archive["parameters"]
            states = archive["states"]
            if parameters.ndim != 2 or parameters.shape[1] != 3:
                raise ValueError("parameters must have shape (trajectories, 3).")
            if (
                states.ndim != 3
                or states.shape[0] != len(parameters)
                or states.shape[1] < 2
                or states.shape[2] != 2
            ):
                raise ValueError("states must have shape (trajectories, at least 2 points, 2).")
            self._parameters = torch.from_numpy(parameters.astype(np.float32, copy=False))
            self._states = torch.from_numpy(states.astype(np.float32, copy=False))
            self._steps = states.shape[1] - 1

    def __len__(self) -> int:
        if self._rows is not None:
            return len(self._rows)
        return len(self._parameters) * self._steps

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rows is not None:
            row = self._rows[index]
            return row[:5], row[5:]

        trajectory, step = divmod(index, self._steps)
        inputs = torch.cat((self._parameters[trajectory], self._states[trajectory, step]))
        return inputs, self._states[trajectory, step + 1]
