from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np
import torch

_log = getLogger(__name__)


class ASRModelBase(ABC):
    model_name: str
    model_dir: Path
    device: str

    @abstractmethod
    def __init__(self, model_name: Path, model_dir: Path):
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_arrays: List[np.ndarray],
        sample_rate: int,
    ) -> List[str]:
        pass
