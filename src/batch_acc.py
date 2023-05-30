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
import json

def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))

# complexity: mid
def get_image_files(path) -> tuple[list, int]:
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Aggiungi qui le estensioni dei file immagine che desideri includere

    image_files = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return image_files, len(image_files)


def eval(device) -> tuple[dict, int, int, list, list]:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    TP_number =  len(get_image_files(os.path.join(current_directory, 'datasets', 'lfw-deepfunneled', 'TP')))
    images, n_images = get_image_files(os.path.join(current_directory, 'datasets', 'lfw-deepfunneled'))
    res = {}
    
    mtcnn = MTCNN(image_size=160, margin=0, select_largest=False, post_process=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    blacklist_embeddings = torch.load('blacklist.pt')
    cont = 1
    
    positive_list = []
    error_list = []
    
    # complexity: 175 embedding db against ~14k images. O(kn) + O(sorting_comp*k)
    for img in images:
        
        print(f"Computing {cont}/{n_images}\n")
        
        input_image = Image.open(img)
        with torch.no_grad():
            input_cropped = mtcnn(input_image.convert("RGB"))
            if(input_cropped is None):
                target='E'
                res[img] = target
                error_list.append(img)
                input_image.close()
                cont+=1
                continue
            input_embedding = model(input_cropped.unsqueeze(0))

        distances = []
        cos = torch.nn.CosineSimilarity(dim=0)

        for features in blacklist_embeddings:
            # print(f"blacklist shape:  {features.shape} \ninput shape: {input_embedding.shape}")
            dist = cos(features, input_embedding.squeeze(0))
            distances.append(dist.item())

        max_distance = max(distances)
        threshold=0.7
        target='F'
        if max_distance >= threshold:
            positive_list.append(img)
            target='T'
            
        res[img] = target    
        
        input_image.close()
        cont +=1
    return res, n_images, TP_number, positive_list, error_list

def main() -> bool:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    r={}
    r['detected'],n, tp_n_images, positive_list, error_list = eval(device)
    counter = Counter(r['detected'].values())
    
    printOut = True
    
    r['positive_targets'] = positive_list
    r['error_targets'] = error_list
    
    print(f"total: {n}")
    print(f"Actual TP: {tp_n_images}")
    print(f"target found: {counter['T']}\ntarget not found: {counter['F']}\ntarget error: {counter['E']}\n")
    
    if printOut:
        print("Positive targets:")
        for e in positive_list:
            print(e)
        print("Error targets:")
        for e in error_list:
            print(e)
    
    with open("out_res.json", "w") as file:
        json.dump(r, file)
    
    
    return True
        
        
if(__name__ == '__main__'):
    main()