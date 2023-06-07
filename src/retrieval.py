import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

class Retrieval():
    _usingMtcnn = False
    _blacklistEmbeddingsFilename = ""
    _blacklistEmbeddings = []
    _distanceThreshold = 0.8
    _distanceFunction = torch.nn.CosineSimilarity(dim=0)
    _debug = False
    _distances = []
    _visualize = False
    _device = None
    
    def __init__(self, embeddingsFileName, usingMtcnn=True, debug=False) -> None:
        self._blacklistEmbeddingsFilename = embeddingsFileName
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._debug = debug
        self._usingMtcnn = usingMtcnn
        if(usingMtcnn):
            self._mtcnn = MTCNN(image_size=160, margin=0, select_largest=False, post_process=True, device=self._device)
        self._model = InceptionResnetV1(pretrained='vggface2').eval()
        try:
            self._blacklistEmbeddings = torch.load(self._blacklistEmbeddingsFilename)
        except:
            print(f"Impossible loading embedding file '{self._blacklistEmbeddingsFilename}'")
            exit(1)

    
    def buildBlacklistEmbeddings(self, blacklistImagesPath) -> None:
        pass
    
    
    def evaluate(self, input_image) -> bool:  
        PILimage = T.ToPILImage(mode='RGB')
        # not needed when passed from opencv and not from file
        if(type(input_image) == np.ndarray):
            input_image = PILimage(input_image)
        #input_image = Image.open(input_image)


        with torch.no_grad():
            if(self._usingMtcnn):
                input_cropped = self._mtcnn(input_image.convert("RGB"))
                if(input_cropped is None):
                    return False
                inference_embedding = self._model(input_cropped.unsqueeze(0))
                if(self._visualize):
                    boxes, probs, landmarks = self._mtcnn.detect(PILimage(input_image), landmarks=True) # type: ignore
                    PILimage(input_cropped).show()
                    fig, ax = plt.subplots(figsize=(16, 12))
                    ax.imshow(PILimage(input_image))
                    ax.axis('off')
                    for box, landmark in zip(boxes, landmarks):
                        ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]])) # type: ignore
                        ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
                    fig.show()
            else:
                inference_embedding = self._model(input_image.convert("RGB"))


        for features in self._blacklistEmbeddings:
            dist = self._distanceFunction(features, inference_embedding.squeeze(0))
            self._distances.append(dist.item())

        max_distance = max(self._distances)
        if(self._debug):
            print(sorted(self._distances))

        if max_distance >= self._distanceThreshold:
            return True
        else:
            return False