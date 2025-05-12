import torch
import math

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from network import Unet
from train import sample
from dataset import get_loader
from torchvision.utils import save_image
from tqdm import tqdm
import os
from torch_fidelity import calculate_metrics


def load_model(path, dataset, res, device):
    checkpoint = torch.load(f"{path}/ddpm_{dataset}_{str(res)}_{str(300)}.pt", map_location=device)
    model = Unet(
        dim=res,
        channels=3,
        dim_mults=(1, 2, 4,)
    )
    model.load_state_dict(checkpoint['model'])
    return model

@torch.no_grad()
def save_real_images(real_dataloader, to_save = 5000):
    os.makedirs("real", exist_ok=True)
    imgs_saved = 0
    for batch in tqdm(real_dataloader, desc= "Saving real images"):
        if imgs_saved >= to_save: 
            break
        imgs, _ = batch
        for img in imgs:
            if imgs_saved >= to_save:
                break
            path = os.path.join("real", f"img_{imgs_saved}.png")
            save_image(img, path, normalize=True)
            imgs_saved+=1
    print("Real images saved.")

def save_generated_images(images, epoch,dataset,res, folder='data'):
    if images.min() < 0:
        images = (images + 1) / 2  # Assuming input is in [-1, 1]

    # Create directory for this epoch
    epoch_dir = os.path.join(folder,f'ddpm_{dataset}_{str(res)}', f'epoch_{str(epoch)}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Save each image
    for i in range(images.size(0)):
        save_path = os.path.join(epoch_dir, f'image_{i:04d}.png')
        save_image(images[i], save_path)
    print(f"Epoch images saved to {epoch_dir}")

def main():
    res = 64
    batch_size= 32
    to_test= 50000
    dataset_name = "CelebA"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("data","CelebA", res, device)
    imgs = sample(model, res, batch_size)
    # print(min(imgs))
    real_loader = get_loader(batch_size, res, dataset_name)
    save_real_images(real_loader, to_save=to_test)

    for i in range(0, to_test, batch_size):
        imgs, w = sample(model,res,batch_size)
        generated_images.append(imgs)
        del imgs, w
    generated_images = torch.cat(generated_images, dim=0)
    save_generated_images(generated_images,"evaluation",dataset_name,res)
    del generated_images
    prc_dict = calculate_metrics(
        input1=f'./real', 
        input2=f'./data/ddpm_{dataset_name}_{str(res)}/epoch_evaluation', 
        cuda=True, 
        isc=False, 
        fid=True, 
        kid=True, 
        prc=True, 
        verbose=False
    )
    inception_dict = calculate_metrics(
        input1=f'./data/ddpm_{dataset_name}_{str(res)}/epoch_evaluation', 
        cuda=True, 
        isc=True, 
        fid=False, 
        kid=False, 
        prc=False, 
        verbose=False
    )
    prc_dict['inception_score_mean'] = inception_dict['inception_score_mean']
    print(prc_dict)
    return
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()    

if __name__=="__main__":
    main()