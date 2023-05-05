import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from facenet_pytorch import MTCNN, InceptionResnetV1

from pprint import pprint

def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


# creo una trasformazione per tornare in PIL Image così da visualizzare le trasformazioni di Image Processing fatte
PILimage = T.ToPILImage(mode='RGB')

# CNN pre-addestrata
mtcnn = MTCNN(image_size=160, margin=0, select_largest=False, post_process=True, device='cuda:0')

model = InceptionResnetV1(pretrained='vggface2').eval()

# carico la blacklist di pytorch
blacklist_embeddings = torch.load('blacklist.pt')

input_image = Image.open('in.jpg')

# boxes, probs, landmarks = mtcnn.detect(input_image, landmarks=True)

# # Visualize
# fig, ax = plt.subplots(figsize=(16, 12))
# ax.imshow(input_image)
# ax.axis('off')

# for box, landmark in zip(boxes, landmarks):
#     ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
#     ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
# fig.show()

with torch.no_grad():
    input_cropped = mtcnn(input_image.convert("RGB"))
    PILimage(input_cropped).show()

    input_embedding = model(input_cropped.unsqueeze(0))

distances = []
cos = torch.nn.CosineSimilarity(dim=0)

for features in blacklist_embeddings:
    #print(t.shape)
    type(features)
    type(input_embedding)
    print(f"blacklist shape:  {features.shape} \ninput shape: {input_embedding.shape}")
    dist = cos(features, input_embedding.squeeze(0))
    distances.append(dist.item())

# Seleziona il volto della lista nera con la distanza max/min
# valutare se usare avg furthest neighbour etc..
max_distance = max(distances)
pprint(sorted(distances))
threshold=0.8

if max_distance >= threshold:
    # Volto nella lista nera trovato
    print('POSITIVE')

else:
    print('NEGATIVE')