from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    PNDMScheduler,
    AutoencoderTiny,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
import torchvision.transforms.functional as TF
from diffusers.models.attention_processor import AttnProcessor2_0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
from collections import deque
from .utils import GridBasedHashTable_Sim

import sys
sys.path.append('./')

from DeepCache import Zero123Pipeline


    
class Zero123_Cache(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers", cache_p=0.5):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.cache_p = cache_p

        self.dtype = torch.float16 if fp16 else torch.float32

        assert self.fp16, 'Only zero123 fp16 is supported for now.'
        
        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)



        # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
        # self.pipe.vae = AutoencoderKL.from_single_file(url, torch_dtype=torch.float16).to("cuda")
       
       # stable-zero123 has a different camera embedding
        self.use_stable_zero123 = 'stable' in model_key
        
        self.pipe.image_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.clip_camera_projection.eval()

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.embeddings = None
        
        
        delta_c = [20, 20, 0.1]  # Different grid sizes for each spatial dimension
        N = [73856093, 19349669, 83492791, 9999953]  # List of constants [N1, N2, N3]
        self.feature_cache = GridBasedHashTable_Sim(delta_c, delta_t=20, N=N, max_queue_length=3, hash_table_size=200)
        # self.prv_features = None

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
        c = self.pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor
        self.embeddings = [c, v]
    def get_cam_embeddings(self, polar, azimuth, radius):
        if self.use_stable_zero123:
            T = np.stack([np.deg2rad(polar), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), np.deg2rad([90-5]*len(polar))], axis=-1)
        else:
            # original zero123 camera embedding
            T = np.stack([np.deg2rad(polar), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype, device=self.device) # [8, 1, 4]
        return T
    
    @torch.no_grad()
    def refine(self, 
               pred_rgb, 
               polar, 
               azimuth, 
               radius, 
               guidance_scale=5, 
               steps=50, 
               strength=0.8,
               cache_interval: int = 1,
               cache_layer_id: int = None,
               cache_block_id: int = None,
               uniform: bool = True,
        ):

        batch_size = pred_rgb.shape[0]

        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn((1, 4, 32, 32), device=self.device, dtype=self.dtype)
            
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        T = self.get_cam_embeddings(polar, azimuth, radius)
        cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
        cc_emb = self.pipe.clip_camera_projection(cc_emb)
        cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

        vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
        vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
        
        if cache_interval == 1:
            interval_seq = list(range(steps))
        else:
            if uniform:
                interval_seq = list(range(0, steps, cache_interval))
            else:
                num_slow_step = steps//cache_interval
                if steps%cache_interval != 0:
                    num_slow_step += 1
                
                interval_seq, pow = sample_from_quad_center(steps, num_slow_step, center=center, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
        
        prv_features = None

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.view(1)] * 2).to(self.device)
            
            if i in interval_seq:
                prv_features = None
            else:
                # print("use cache")
                pass
            
            
            # predict the noise residual
            noise_pred, prv_features = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
                replicate_prv_feature=prv_features,
                quick_replicate= cache_interval>1,
                cache_layer_id=cache_layer_id,
                cache_block_id=cache_block_id,
                return_dict=False,
            )
            


            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 256, 256]
        return imgs
    
    def train_step(self, 
                pred_rgb, 
                polar, 
                azimuth, 
                radius, 
                step_ratio=None, 
                guidance_scale=5, 
                as_latent=False,
                cache=False,
                cache_layer_id=0,
                cache_block_id=0,
                noise=None):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        
        
        # interval_seq = list(range(self.min_step, self.max_step + 1, cache_interval))
        with torch.no_grad():
            if noise is None:
               noise = torch.randn_like(latents)

            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            T = self.get_cam_embeddings(polar, azimuth, radius)
            cc_emb = torch.cat([self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
            cc_emb = self.pipe.clip_camera_projection(cc_emb)
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

            vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
            
            noise_pred = self.forward_unet(x_in, 
                                           vae_emb, 
                                           t, t_in,
                                           cc_emb, 
                                           polar, azimuth, radius, 
                                           cache=cache, 
                                           cache_layer_id=cache_layer_id, 
                                           cache_block_id=cache_block_id)
            


        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')

        return loss
    
    def forward_unet(self,
                    x_in,
                    vae_emb,
                    t,
                    t_in,
                    cc_emb,
                    polar,
                    azimuth,
                    radius,
                    cache=True,
                    cache_layer_id=0,
                    cache_block_id=0):
        
        
        polar = torch.tensor(polar).to(self.device)
        azimuth = torch.tensor(azimuth).to(self.device)
        radius = torch.tensor(radius).to(self.device)
        
        batch_size = len(polar)  # Determine batch size from elevation tensor shape
        prv_features = None  # Initialize variable to hold previously cached noise
        key = torch.stack([t[:batch_size], polar, azimuth, radius], dim=-1)

        # Random chance to update cached noise
        if random.random() < self.cache_p:
            
            has_none = False  # Flag to check if any noise is missing in the cache
            
            # Iterate through each batch item and query cached noise
            for i in range(batch_size):
                prv_feature = self.feature_cache.query(key[i], x_in[i])
                if prv_feature is None:
                    has_none = True
                    break
                else:
                    # Update previously cached noise
                    self.prv_features[i] = prv_feature[0]
                    self.prv_features[i + batch_size] = prv_feature[1]
                    
            # Decide whether to use the cached features based on their availability
            prv_features = None if has_none else self.prv_features
       
        # Flag to determine if new features need to be appended to the cache
        append = prv_features is None
                
        noise_pred, prv_features = self.unet(
            torch.cat([x_in, vae_emb], dim=1),
            t_in.to(self.unet.dtype),
            encoder_hidden_states=cc_emb,
            replicate_prv_feature=prv_features,
            quick_replicate=cache,
            cache_layer_id=cache_layer_id,
            cache_block_id=cache_block_id,
            return_dict=False,
        )
        
        if append:
            # Update cache with new noise predictions
            for i in range(batch_size):
                t_features = torch.stack([prv_features[i], prv_features[i + batch_size]], dim=0)
                self.feature_cache.append(key[i], x_in[i], t_features)
                
            self.prv_features = prv_features
        
        return noise_pred

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    
if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import kiui
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('--polar', type=float, default=0, help='delta polar angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0, help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0, help='delta camera radius multiplier in [-0.5, 0.5]')
    parser.add_argument('--stable', action='store_true')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = kiui.read_image(opt.input, mode='tensor')
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)

    print(f'[INFO] loading model ...')
    if opt.stable:
        zero123 = Zero123_Cache(device, model_key='ashawkey/stable-zero123-diffusers')
    else:
        zero123 = Zero123_Cache(device, model_key='ashawkey/zero123-xl-diffusers')

    zero123 = Zero123_Cache(device)

    print(f'[INFO] running model ...')
    zero123.get_img_embeds(image)

    while True:
        start_time = time.time()
        outputs = zero123.refine(image, polar=[opt.polar], azimuth=[opt.azimuth], radius=[opt.radius], 
                                 strength=0, 
                                 steps=50, 
                                 cache_interval=5, 
                                 cache_layer_id=0, 
                                 cache_block_id=0)
        
        refine_use_time = time.time() - start_time
        print("Cache - Refiner: {:.2f} seconds".format(refine_use_time))
        plt.imshow(outputs.float().cpu().numpy().transpose(0, 2, 3, 1)[0])
        # plt.show()
        plt.savefig('test_cache.png')
        exit()

