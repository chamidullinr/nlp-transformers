import random
import os
import numpy as np
import torch


def set_random_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_progress_bar():
    import transformers
    import datasets

    # disable progress bar
    datasets.logging.get_verbosity = lambda: datasets.logging.NOTSET
    transformers.logging.get_verbosity = lambda: transformers.logging.NOTSET
