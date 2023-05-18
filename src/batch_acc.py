import torch
import os
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from collections import Counter
from pprint import pprint

def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))

# complexity: high
def get_image_files(path) -> list:
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Aggiungi qui le estensioni dei file immagine che desideri includere

    image_files = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return image_files


def eval() -> dict:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    images = get_image_files(os.path.join(current_directory, 'datasets', 'lfw-deepfunneled', 'lfw'))
    res = {}
    
    mtcnn = MTCNN(image_size=160, margin=0, select_largest=False, post_process=True, device='cuda:0')
    model = InceptionResnetV1(pretrained='vggface2').eval()
    blacklist_embeddings = torch.load('blacklist.pt')

    # complexity: 175 embedding db against ~14k images. O(kn) + O(sorting_comp*k)
    for img in images:
        input_image = Image.open(img)

        with torch.no_grad():
            input_cropped = mtcnn(input_image.convert("RGB"))
            input_embedding = model(input_cropped.unsqueeze(0))

        distances = []
        cos = torch.nn.CosineSimilarity(dim=0)

        for features in blacklist_embeddings:
            # print(f"blacklist shape:  {features.shape} \ninput shape: {input_embedding.shape}")
            dist = cos(features, input_embedding.squeeze(0))
            distances.append(dist.item())

        max_distance = max(distances)
        threshold=0.8
        target=False
        if max_distance >= threshold:
            target=True
            
        res[img] = target    
        
        input_image.close()
    return res

def main() -> bool:
    r = eval()
    counter = Counter(r.values())
    
    print(f"target found: {counter[True]}\ntarget not found: {counter[False]}" )
    
    
    return True
        
        
if(__name__ == '__main__'):
    main()