import torch
import math

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from network import Unet
from train import sample

def load_model(path, dataset, res, device):
    checkpoint = torch.load(f"{path}/ddpm_{dataset}_{str(res)}_{str(300)}.pt", map_location=device)
    model = Unet(
        dim=res,
        channels=3,
        dim_mults=(1, 2, 4,)
    )
    model.load_state_dict(checkpoint['model'])
    return model

def main():
    res = 64
    batch_size= 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("data","CelebA", res, device)
    imgs = sample(model, res, batch_size)
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()    

if __name__=="__main__":
    main()