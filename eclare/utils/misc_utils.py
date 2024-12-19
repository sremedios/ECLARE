import torch
from tqdm.auto import tqdm
from functools import wraps


def parse_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        print("GPU index not provided or no GPU support currently available.")
        print("!!! Running on CPU !!!")
        device = torch.device("cpu")
    return device


class LossProgBar:
    def __init__(self, total, update_amount, loss_names, precision=4):
        self.total = total
        self.update_amount = update_amount
        self.precision = precision
        self.pbar = tqdm(total=self.total)
        self.pbar_dict = {name: torch.finfo(torch.float32).max for name in loss_names}

    def __enter__(self):
        self.pbar = tqdm(total=self.total)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.close()

    def close(self):
        self.pbar.close()

    def update(self, loss_dict):
        for k, v in loss_dict.items():
            self.pbar_dict[k] = v.detach().cpu().numpy().item()
        self.pbar.set_postfix(
            {k: f"{v:.{self.precision}f}" for k, v in self.pbar_dict.items() if v != 0}
        )
        self.pbar.update(self.update_amount)
