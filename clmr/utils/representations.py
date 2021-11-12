import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple


def extract_representations(
    dataloader: DataLoader,
    encoder: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    representations = []
    ys = []
    for x, y in tqdm(dataloader):
        with torch.no_grad():
            x = x.to(device)
            h0 = encoder(x)
            representations.append(h0)
            ys.append(y)

    if len(representations) > 1:
        representations = torch.cat(representations, dim=0)
        ys = torch.cat(ys, dim=0)
    else:
        representations = representations[0]
        ys = ys[0]
    return representations, ys


def save_representations(fp: str, representation: torch.Tensor, y: torch.Tensor):
    torch.save(
        {
            "representation": representation,
            "y": y,
        },
        fp,
    )
