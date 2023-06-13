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
from torch.nn import CosineSimilarity


from imageaugmenter import ImageAugmenter


class Retrieval():
    _usingMtcnn:bool = False
    _blacklistEmbeddingsFilename:str = ""
    _blacklistEmbeddings:list[float] = []
    _distanceThreshold:float = 0
    def _distanceFunction(x,y): torch.cdist(x,y,2) # type: ignore
    _debug:bool = False
    _distances:list[float] = []
    _visualize:bool = False
    _device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _blacklistFolderName:str = ""
    _workspacePath:str = getcwd()
    _weigths:str = ""
    _usingAverage:bool = True
    _usingMedian:bool = False
    _usingMax:bool = False
    _distanceMetric:str = "L2"
    _status:int = 0
    
    def __init__(self, embeddingsFileName, weights='vggface2', threshold=0.1, distanceMetric='cosine', usingMedian = False, usingMax=False, usingMtcnn=True, usingAverage = True, toVisualize=True, debug=False) -> None:
        """constructor of the class

        Args:
            embeddingsFileName (_type_): .pt file with the blacklist embeddings.
            weights (str, optional): pretrained weights of facenet. Defaults to 'vggface2'.
            threshold (float, optional): threshold value that determines the final label, be careful changing this accordingly the metrics you chose. Defaults to 0.7.
            distanceMetric (str, optional): distance metric between embeddings. Defaults to 'L2'.
            usingMedian (bool, optional): if element to group distance is given by median. Defaults to False.
            usingMax (bool, optional): if element to group distance is given by max. Defaults to False.
            usingMtcnn (bool, optional): if pre-processing mctnn is active. Defaults to True.
            usingAverage (bool, optional): if element to group distance is given by average. Defaults to True.
            toVisualize (bool, optional): if you wish to visualize resized and prewhitened image. Defaults to False.
            debug (bool, optional): enables debug. Defaults to False.
        """
        self._distanceThreshold = threshold
        self._usingMax = usingMax
        self._usingAverage = usingAverage
        self._usingMedian = usingMedian
        self._visualize = toVisualize
        self._blacklistEmbeddingsFilename = embeddingsFileName
        self._weigths = weights
        self._debug = debug
        self._usingMtcnn = usingMtcnn
        self._distanceMetric = distanceMetric
        self.toPilImage = T.ToPILImage(mode='RGB')
        
        self._Augmenter = ImageAugmenter()
        
        
        if(sum([usingAverage, usingMedian, usingMax]) > 1):
            print("You can't use more than one method to compare the embeddings, please choose only one.")
            exit(1)
        
        if(sum([usingAverage, usingMedian, usingMax]) == 0):
            self._usingMax = True
            print("Using max as default method to compare the embeddings.")
        
        if(usingMtcnn):
            self._mtcnn = MTCNN(image_size=160, margin=0, select_largest=False, post_process=True, device=self._device)
        self._model = InceptionResnetV1(pretrained=self._weigths).eval()
        try:
            self._blacklistEmbeddings = torch.load(self._blacklistEmbeddingsFilename)
        except:
            print(f"Impossible to load pytorch embedding file, remember to build one before.\n Filename: '{self._blacklistEmbeddingsFilename}'")
            exit(1)


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
    
    def evalFrameTextual(self, input_image) -> tuple[int,str]:
        ret = self.__evaluateFrame(input_image)
        self._status = ret
        return ret, self.__getTextualOutputFromResult()
    
    def evalFrame(self, input_image) -> int:
        ret = self.__evaluateFrame(input_image)
        self._status = ret
        return ret
    
    
    def __evaluateFrame(self, input_image) -> int:
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
            if self._distanceMetric == "L2":
                features = features.unsqueeze(0)        # features: 1*512
                #inference_embedding = inference_embedding.squeeze(0) # input_embedding: 1*512
                dist = torch.cdist(features, inference_embedding, 2) # input_embedding: 1*512
                self._distances.append(dist.item())

            if self._distanceMetric == "cosine":
                # use cosine similarity
                cos = CosineSimilarity(dim=1, eps=1e-6)
                features = features.unsqueeze(0)        # features: 1*512
                d = cos(features, inference_embedding)
                self._distances.append(d.item())

        max_distance = max(self._distances)
        avg_distance:float = sum(self._distances)/len(self._distances)
        median_distance = np.median(self._distances)
        
        if self._usingAverage:
            distance = avg_distance
        else:
            if self._usingMedian:
                distance = median_distance
            else:
                distance = max_distance
        
        if(self._debug):
            if(self._usingAverage):
                print(f"Average distance: {distance}")
            if(self._usingMedian):
                print(f"Median distance: {distance}")
            if(self._usingMax):
                print(f"Max distance: {distance}")
                
        if distance >= self._distanceThreshold:
            return 1
        else:
            return 2
                
        
    # testing
    
    def __get_image_files(self, path) -> tuple[list, int]:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

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


    # getters and setters

    def isUsingMtcnn(self) -> bool:
        return self._usingMtcnn
    
    def hasMtcnnFailed(self) -> bool:
        return self._status == 3 and self.isUsingMtcnn()
    
    def isPersonBlacklisted(self) -> bool:
        return self._status == 1

    def getStatus(self) -> int:
        return self._status

    def setDistanceThreshold(self, threshold):
            self._distanceThreshold = threshold
        
    def setDistanceFunction(self, distanceFunction):
            self._distanceFunction = distanceFunction
            
    def __getTextualOutputFromResult(self) -> str:
        retrieval_label = {1:'Detected and identified', 2:'Detected but not identified', 3: 'Not available yet',}
        return retrieval_label[self._status]