import torch
import numpy as np
import torch.nn.functional as F
from schedules import linear_beta_schedule
from tqdm import tqdm
from network import Unet
from network import num_to_groups
from torch.optim import Adam
from dataset import get_loader
import matplotlib.pyplot as plt
from einops import rearrange
import time
from util import compute_kid, load_real_images, save_metrics, save_model, compute_fid, save_generated_images
import torchvision.utils as vutils


timesteps=300

if __name__=="__main__":
    timesteps = 300
    betas = linear_beta_schedule(timesteps=timesteps)
    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    #print("betas initialized")
    img_res = 64
    batch_size=32
    channels=3
    dataset_name = "CelebA"
    device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    model = Unet(
        dim=img_res,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 5
    kid_samples = 500
    data_loader = get_loader(batch_size,img_res,dataset_name)
    """
    single_img, _ = next(iter(data_loader))
    
    t = torch.randint(50, 100, (1,), device=device).long()
    noised = q_sample(x_start=single_img, t=t)
    single_img = single_img[0]
    noised = noised[0]
    single_img = (single_img + 1) / 2
    noised = (noised +1 )/2
    print(f"img shape: {single_img.shape}")
    single_img = rearrange(single_img, 'c h w -> h w c')
    noised = rearrange(noised, 'c h w -> h w c')
    print(f"img shape: {single_img.shape}")
    print("finished dataloader")
    plt.figure()
    plt.imshow(single_img)
    plt.axis('off')
    plt.show()
    plt.figure()
    plt.imshow(noised)
    plt.axis('off')
    plt.show()
    """
    gpu_mb_alloc = []
    gpu_mb_reserved = []
    times_per_epoch = []
    total_imgs_seen = 0
    losses = []
    #best_model = model.state_dict()
    #best_kid = 1000
    true_images = load_real_images(data_loader,device, kid_samples)
    kid_means = []
    kid_stds = []
    fid_scores = []
    best_fid = 10000
    best_kid = 10
    best_fid_model = model.state_dict()
    best_kid_model = model.state_dict()
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for batch_idx, (imgs, _) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()

            imgs = imgs.to(device)
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, imgs, t)

            if batch_idx % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            gpu_mb_alloc.append(torch.cuda.memory_allocated() / (1024 ** 2))    # potentially track every few batches rather than every batch
            gpu_mb_reserved.append(torch.cuda.memory_reserved() / (1024 ** 2))
            total_imgs_seen+=batch_size
        times_per_epoch.append(time.time()-start_time)

        # compute KID
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            generated_images = []
            num_batches = kid_samples // batch_size # there is going to be slight imbalance (5000 real vs 5024 fake) but should not affect results and still yeild in accurate result
            for _ in range(num_batches):
                gen_images = sample(model, img_res, batch_size, channels)
                generated_images.append(gen_images)
            generated_images = torch.cat(generated_images, dim=0)
            kid_mean, kid_std = compute_kid(true_images,generated_images, device, res = img_res, batch_size=batch_size, sample_size=500)
            fid_score = compute_fid(true_images, generated_images,device)
            kid_means.append(kid_mean)
            kid_stds.append(kid_std)
            fid_scores.append(fid_score)
            save_generated_images(generated_images,epoch, dataset_name, img_res, timesteps)
        del generated_images
        torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{epochs}: KID - {kid_mean}, FID - {fid_score}")
        if fid_score < best_fid:
            best_fid = fid_score
            best_fid_model = model.state_dict()
        if kid_mean < best_kid:
            best_kid = kid_mean
            best_kid_model = model.state_dict()

    time_per_kimg = ((sum(times_per_epoch)/len(times_per_epoch))/(len(data_loader)*batch_size))*1000
    cum_times = np.cumsum(np.array(times_per_epoch))
            #del t
            #torch.cuda.empty_cache
            # save generated images
    print(f"Time per 1 kimg: {time_per_kimg:.2f}")
    #print(f"KID means: {kid_means}")
    #print(f"KID stds: {kid_stds}")
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")
    plt.savefig(f"loss_plot_ddpm_{dataset_name}_{str(img_res)}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.title("Training Loss (log) Curve")
    plt.savefig(f"loss_plot_ddpm_{dataset_name}_{str(img_res)}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(gpu_mb_alloc, label="CUDA Memory Allocated (MB)")
    plt.plot(gpu_mb_reserved, label="CUDA Memory Reserved (MB)")
    plt.xlabel("Iteration")
    plt.ylabel("MB")
    plt.legend()
    plt.title("Training Memory Usage")
    plt.savefig(f"memory_plot_ddpm_{dataset_name}_{str(img_res)}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    x = range(1,epochs+1)
    plt.errorbar(x,kid_means,yerr=kid_stds,capsize=5, label="KID scores")
    plt.xlabel("Epoch")
    plt.ylabel("KID")
    plt.legend()
    plt.title("Training KID curve")
    plt.savefig(f"kid_plot_ddpm_{dataset_name}_{str(img_res)}_{str(timesteps)}.png")
    plt.show()

    fig, ax1 = plt.subplots()   # may need to fix figure size to (10,5) too

    # FID line (left y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('FID', color=color)
    ax1.plot(cum_times, fid_scores, color=color, label='FID')
    ax1.tick_params(axis='y', labelcolor=color)

    # KID line with error bars (right y-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('KID', color=color)
    ax2.errorbar(cum_times, kid_means, yerr=kid_stds, color=color, linestyle='--', marker='o', label='KID ± std')
    ax2.tick_params(axis='y', labelcolor=color)

    # Layout and title
    fig.tight_layout()
    plt.title("FID and KID over Training Time")
    plt.savefig(f"fid_vs_kid_ddpm_plot_{dataset_name}_{str(img_res)}.png")
    plt.show()
    """if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)"""
    save_metrics("data",cum_times, kid_means, kid_stds, gpu_mb_alloc, gpu_mb_reserved, time_per_kimg, batch_size, dataset_name, img_res, timesteps)
    save_model("data", model,best_fid_model,best_kid_model, dataset_name, img_res, timesteps)
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    loss = F.mse_loss(noise, predicted_noise)
    return loss

@torch.no_grad()
def generate_images(model, noise=None): # used for calculating FID/KID and other metrics
    model.eval()
    if noise is None:
        noise = torch.randn(batch_size, channels, img_res, img_res)
    print(noise.shape)

    
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 



# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    #imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        #imgs.append(img.cpu().numpy())
    return img.detach() 

@torch.no_grad()
def sample(model, image_size, batch_size=32, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def load_model(path, dataset, res, device):
    checkpoint = torch.load(f"{path}/ddpm_{dataset}_{str(res)}_{str(300)}.pt", map_location=device)
    model = Unet(
        dim=res,
        channels=3,
        dim_mults=(1, 2, 4,)
    )
    model.load_state_dict(checkpoint['model'])
    return model

def eval_model():
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
    train()