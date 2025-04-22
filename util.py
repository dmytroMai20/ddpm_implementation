import torch
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import math

transform = T.Compose([
    T.Resize((299, 299)),  # InceptionV3 expects 299x299
    T.Normalize([-1]*3, [2]*3)
])
@torch.no_grad()
def load_real_images(real_dataloader, device, sample_size= 5000): # fit all real images on the device straight away to reduce I/O cost of fitting on device 
    real_images = []
    for imgs, _ in real_dataloader:
        real_images.append(imgs)
        if sum([i.size(0) for i in real_images]) >= sample_size:
            break
    real_images = torch.cat(real_images, dim=0)[:sample_size]
    real_images = transform(real_images).to(device)
    return real_images

@torch.no_grad()
def compute_kid(real_imgs, fake_imgs, device, res=64, batch_size=32, sample_size=500): # smaller sample size for kid
    #real_imgs.to(device) # should already be on device and transformed from load_real_images
    #real_imgs = transform(real_imgs)
    fake_imgs = transform(fake_imgs)
    kid = KernelInceptionDistance(feature=2048,subset_size=50, normalize=True).to(device)
    kid.update(real_imgs[:sample_size], real=True)
    kid.update(fake_imgs[:sample_size], real=False)
    #del fake_imgs
    #torch.cuda.empty_cache
    kid_values = kid.compute()
    kid.reset()
    return [kid_values[0].item(), kid_values[1].item()]

def save_metrics(path, times, kids_mean, kids_stds, gpu_alloc, gpu_reserved, time_kimg, batch_size, dataset, res, max_t):
    save_path = f"{path}/ddpm_{dataset}_{str(res)}_{str(max_t)}_metrics.pth"
    torch.save({'times':times,
                'kids_mean':kids_mean,
                'kids_stds':kids_stds,
                'gpu_alloc':gpu_alloc,  #gpu alloc and reserved in mb
                'gpu_reserved':gpu_reserved,
                'time_kimg':time_kimg,
                'batch_size':batch_size}, save_path)
    
def save_model(path, model, dataset,res, t_max):
    save_path = f"{path}/ddpm_{dataset}_{str(res)}_{str(t_max)}.pt"
    torch.save({'model':model.state_dict()
                 }, save_path)