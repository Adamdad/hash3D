# eval the clip-similarity for an input image and a generated images
import os
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
import tqdm

# openai/clip-vit-large-patch14
# openai/clip-vit-base-patch32
# laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
class CLIP:
    def __init__(self, device, model_name='openai/clip-vit-large-patch14'):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # normalize features
        return image_features

    def encode_text(self, text):
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # normalize features
        return text_features
    
def sample_images(folder, num_images):
    all_images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sampled = np.linspace(0, len(all_images) - 1, num_images, dtype=int)
    return [all_images[i] for i in sampled]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help="path to front view image")
    parser.add_argument('folder', type=str, help="path to folder containing rendered images")
    parser.add_argument('--mode', type=str, choices=["image", "text"], default="image", help="mode to use for CLIP")
    parser.add_argument('--num_images', type=int, default=8, help="number of images to sample from the folder")
    parser.add_argument('--model_name', type=str, default='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', help="CLIP model name")
    
    opt = parser.parse_args()

    # Initialize CLIP
    clip = CLIP('cuda', model_name=opt.model_name)
    if opt.mode == "image":
        # Load reference image and encode as reference features
        ref_img = cv2.imread(opt.image, cv2.IMREAD_COLOR)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            ref_features = clip.encode_image(ref_img)
    elif opt.mode == "text":
        ref_text = opt.image
        with torch.no_grad():
            ref_features = clip.encode_text(ref_text)
        
    # Sample images from the folder
    sampled_images = sample_images(opt.folder, opt.num_images)

    # Iterate through sampled images
    results = []
    for filename in tqdm.tqdm(sampled_images):
        image_path = os.path.join(opt.folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            cur_features = clip.encode_image(image)
        similarity = (ref_features * cur_features).sum(dim=-1).mean().item()
        results.append(similarity)
    
    # Compute the average similarity
    avg_similarity = np.mean(results)
    print(f'Average similarity: {avg_similarity}')
