import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds to ensure reproducibility

    Parameters:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Set deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
