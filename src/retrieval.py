import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from numpy import asarray
from os import listdir, getcwd, walk
from os.path import join
from json import dump
from collections import Counter
from cv2 import resize, cvtColor, imshow, COLOR_BGR2RGB, COLOR_RGB2BGR, INTER_CUBIC, INTER_AREA, INTER_LINEAR

class Retrieval():
    _usingMtcnn:bool = False
    _blacklistEmbeddingsFilename:str = ""
    _blacklistEmbeddings:list = []
    _distanceThreshold:float = 0
    def _distanceFunction(x,y): torch.cdist(x,y,2) # type: ignore
    _debug:bool = False
    _distances:list[float] = []
    _visualize:bool = False
    _device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _blacklistFolderName:str = ""
    _workspacePath:str = getcwd()
    _weigths:str = ""
    _debugAverage:bool = True
    _usingAverage:bool = True
    
    def __init__(self, embeddingsFileName, weights='vggface2', threshold=0.7, usingMtcnn=True, usingAverage = True, toVisualize=False, debug=False, debugAverage=True) -> None:
        self._distanceThreshold = threshold
        self._debugAverage = debugAverage
        self._usingAverage = usingAverage
        self._visualize = toVisualize
        self._blacklistEmbeddingsFilename = embeddingsFileName
        self._weigths = weights
        self._debug = debug
        self._usingMtcnn = usingMtcnn
        self.toPilImage = T.ToPILImage(mode='RGB')
        if(usingMtcnn):
            self._mtcnn = MTCNN(image_size=160, margin=0, select_largest=False, post_process=True, device=self._device)
        self._model = InceptionResnetV1(pretrained=self._weigths).eval()
        try:
            self._blacklistEmbeddings = torch.load(self._blacklistEmbeddingsFilename)
        except:
            print(f"Impossible to load pytorch embedding file, remember to build one before.\n Filename: '{self._blacklistEmbeddingsFilename}'")

    def setDistanceThreshold(self, threshold):
            self._distanceThreshold = threshold
        
    def setDistanceFunction(self, distanceFunction):
            self._distanceFunction = distanceFunction

    # building
    def buildBlacklistEmbeddings(self, blacklistFolderName="blacklist", augmentation_iter=1) -> None:
        self._blacklistFolderName = join(self._workspacePath, 'src', blacklistFolderName)
        
        self._blacklistEmbeddings = []
        
        blacklist_images = listdir(self._blacklistFolderName)
        blacklist_images = [self._blacklistFolderName + '/' + e for e in blacklist_images]

        process = 0
        
        for image_path in blacklist_images:
            currentImage = Image.open(image_path)
            croppedImage = self._mtcnn(currentImage)
            
            with torch.no_grad():
                img_embedding = self._model(croppedImage.unsqueeze(0))
                
            self._blacklistEmbeddings.extend(img_embedding)
            if(self._debug):
                print(f" --> {process} : {len(blacklist_images)}")
            process += 1

        torch.save(self._blacklistEmbeddings, self._blacklistEmbeddingsFilename)
    
    
    # inference
    
    
    def __prewhiten(self, x):
        mean = x.mean()
        std = x.std()
        std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
        y = (x - mean) / std_adj
        return y
    
    def evaluateFrame(self, input_image) -> int:
        """_summary_

        Args:
            input_image (PIL, ndarray etc): current frame to evaluate

        Returns:
            int: 4 result not yet avialable, 3 if no face detected, 2 if face detected but not recognized, 1 if face detected and recognized
        """
        self._distances.clear()
        
        if(type(input_image) == np.ndarray):
            input_image = self.toPilImage(input_image)


        with torch.no_grad():
            if(self._usingMtcnn):   # 3*160*160
                input_cropped = self._mtcnn(input_image.convert("RGB"))
                if(input_cropped is None):
                    return 3        # fallback
                inference_embedding = self._model(input_cropped.unsqueeze(0))
                if(self._visualize):
                    boxes, probs, landmarks = self._mtcnn.detect(input_image, landmarks=True) # type: ignore
                    self.toPilImage(input_cropped).show()
                    fig, ax = plt.subplots(figsize=(16, 12))
                    adj_img = cvtColor(np.array(input_image), COLOR_BGR2RGB)    
                    ax.imshow(adj_img)
                    ax.axis('off')
                    for box, landmark in zip(boxes, landmarks):
                        ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]])) # type: ignore
                        ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
                    fig.show()
            else:
                #x = asarray(resized.convert("RGB")) # 480*480*3
                x = asarray(input_image.convert("RGB"))
                resized = resize(x, dsize=(160,160), interpolation=INTER_LINEAR)
                if(self._visualize):
                    imshow('resized',resized) # type: ignore

                y = torch.Tensor(resized).permute(2,0,1) # 3*160*160
                y = y.unsqueeze(0) # 1*3*160*160
                y = self.__prewhiten(y)
                
                if(self._visualize):
                    processed = y.squeeze(0).permute(1,2,0).numpy()
                    imshow('prewhitened', processed)
                inference_embedding = self._model(y)


        for features in self._blacklistEmbeddings:
            dist = torch.dist(features, inference_embedding, 2)
            self._distances.append(dist.item())

        max_distance = max(self._distances)
        avg_distance:float = sum(self._distances)/len(self._distances)
        
        if self._usingAverage: distance = avg_distance
        else: distance = max_distance
        
        if(self._debug):
            if(self._debugAverage and self._usingAverage):
                print(f"Average distance: {avg_distance}")
            else:
                print(sorted(self._distances))
                
        if distance <= self._distanceThreshold:
            return 1
        else:
            return 2
                
        
    # testing
    
    def __get_image_files(self, path) -> tuple[list, int]:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Aggiungi qui le estensioni dei file immagine che desideri includere

        image_files = []

        for root, dirs, files in walk(path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(join(root, file))

        return image_files, len(image_files)
    
    def __computeAccuracySide(self, testSetPath, truePositivePath) -> tuple[dict, int, int, list, list]:
        _, TP_number =  self.__get_image_files(truePositivePath)
        images, n_images = self.__get_image_files(testSetPath)
        res = {}
        blacklist_embeddings = torch.load(self._blacklistEmbeddingsFilename)
        cont = 1

        positive_list = []
        error_list = []
        distances = []

        for img in images:
            if(self._debug):
                print(f"Computing {cont}/{n_images}\n")

            input_image = Image.open(img)
            with torch.no_grad():
                input_cropped = self._mtcnn(input_image.convert("RGB"))
                if(input_cropped is None):
                    target='E'
                    res[img] = target
                    error_list.append(img)
                    input_image.close()
                    cont+=1
                    continue
                input_embedding = self._model(input_cropped.unsqueeze(0))

            for features in blacklist_embeddings:
                dist = self._distanceFunction(features, input_embedding.squeeze(0))
                distances.append(dist.item())

            max_distance = max(distances)
            target='F'
            if max_distance >= self._distanceThreshold:
                positive_list.append(img)
                target='T'

            res[img] = target    

            input_image.close()
            cont += 1
        return res, n_images, TP_number, positive_list, error_list



    def computeAccuracy(self, testSetPath, truePositivePath, stdoutResult=True, resultsFileName="results_accuracy.json") -> None:
        resultDictionary = {}
        resultDictionary['detected'],n, tp_n_images, positive_list, error_list = self.__computeAccuracySide(testSetPath, truePositivePath)
        counter = Counter(resultDictionary['detected'].values())

        resultDictionary['positive_targets'] = positive_list
        resultDictionary['error_targets'] = error_list

        print(f"Total: {n}")
        print(f"Actual TP: {tp_n_images}")
        print(f"target found: {counter['T']}\ntarget not found: {counter['F']}\ntarget error: {counter['E']}\n")

        if stdoutResult:
            print("Positive targets:")
            for e in positive_list:
                print(e)
            print("Error targets:")
            for e in error_list:
                print(e)

        with open(resultsFileName, "w") as file:
            dump(resultDictionary, file)


    def isUsingMtcnn(self) -> bool:
        return self._usingMtcnn