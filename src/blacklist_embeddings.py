import torch
import torchvision.transforms as T
from PIL import Image
from sys import path
from os import listdir
from facenet_pytorch import MTCNN, InceptionResnetV1

# model loading
print("Pytorch CUDA Version is ", torch.version.cuda)

batch_cont = 0

if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

BLACKLIST_FOLDER_NAME = 'blacklist'
AUGMENTATION_ITER = 5


model = InceptionResnetV1(pretrained='vggface2').eval()


# creo una trasformazione per tornare in PIL Image così da visualizzare le trasformazioni di Image Processing fatte
PILimage = T.ToPILImage(mode='RGB')

# pipeline con MTCNN
mtcnn = MTCNN(image_size=160, margin=0)

workspace_path = str(path)

# funzione per ottenere il vettore di embedding di una data immagine di volto
def get_face_embedding(image_path):
    image = Image.open(image_path)
    
    img_cropped = mtcnn(image)
    
    PILimage(img_cropped).show()
    
    with torch.no_grad():
        img_embedding = model(img_cropped.unsqueeze(0))
    
    return img_embedding

# definisci una lista di immagini di volti nella lista nera
# viene considerata ogni img nella cartella blacklist
blacklist_images = listdir(BLACKLIST_FOLDER_NAME)
blacklist_images = [BLACKLIST_FOLDER_NAME + '/' + e for e in blacklist_images]


# genera i vettori di embedding per ogni immagine nella lista nera
blacklist_embeddings = []
for image_path in blacklist_images:
    embedding = get_face_embedding(image_path)
    blacklist_embeddings.extend(embedding) # extend siccome ritorno una lista e voglio che i suoi elementi vadano in questa, non voglio append di liste come iter
    print(f" --> {batch_cont} : {len(blacklist_embeddings)}")
    batch_cont+=1

# salva i vettori di embedding della lista nera in un file .pt
#torch.save(blacklist_embeddings, 'blacklist.pt')

def main():
    pass


if (__name__ == '__main__'):
    main()