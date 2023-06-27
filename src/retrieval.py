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
from cv2 import cvtColor, imshow, COLOR_BGR2RGB, COLOR_RGB2BGR, INTER_CUBIC, INTER_AREA, INTER_LINEAR
from torch.nn import CosineSimilarity
from imutils import resize as imresize

from .image_augmenter import ImageAugmenter

class Metrics():
    _precision:float = 0
    _recall:float = 0
    _f1_score:float = 0
    _f1_computed:bool = True
    def __init__(self, precision:float, recall:float, computeF1=True) -> None:
        self._precision = precision * 100
        self._recall = recall * 100
        self._f1_computed = computeF1
        self.__computeF1Score()
    
    def __computeF1Score(self):
        self._f1_score = 2 * ((self._precision * self._recall) / (self._precision + self._recall))
        
    def getPrecision(self):
        return self._precision
    def getRecall(self):
        return self._recall
    def getF1Score(self):
        return self._f1_score
    def isF1Computed(self):
        return self._f1_computed


class Retrieval():
    _usingMtcnn:bool = False
    _blacklistEmbeddingsFilename:str = ""
    _blacklistEmbeddings:list[float] = []
    _distanceThreshold:float = 0
    _debug:bool = False
    _distances:list[float] = []
    _visualize:bool = False
    _device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _blacklistFolderName:str = ""
    _workspacePath:str = getcwd()
    _weights:str = ""
    _usingAverage:bool = True
    _usingMedian:bool = False
    _usingMax:bool = False
    _distanceMetric:str = ""
    _status:int = 0
    _imagesCap:int = 0
    _augmenter:ImageAugmenter = ImageAugmenter(usingSuggestedTransforms=True)
    _augmentationIter:int = 0
    _areEmbeddingsAugmented:bool = False
    _mtcnnShowLandmarksPostProcessing:bool = False
    
        
    
    
    
    def __init__(self, embeddingsFileName, weights='vggface2', threshold=0.1, distanceMetric='cosine', imagesCap=0, usingMedian = False, usingMax=False, usingMtcnn=True, usingAverage = True, toVisualize=True, augmentation_iter=0, mtcnnShowLandmarksPostProcess=False, debug=False) -> None:
        """constructor of the class

        Args:
            embeddingsFileName (_type_): .pt file with the blacklist embeddings.
            weights (str, optional): pretrained weights of facenet. Defaults to 'vggface2', alternative: 'casia-webface'
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
        self._weights = weights
        self._debug = debug
        self._usingMtcnn = usingMtcnn
        self._distanceMetric = distanceMetric
        self.toPilImage = T.ToPILImage(mode='RGB')
        self._mtcnnShowLandmarksPostProcessing = mtcnnShowLandmarksPostProcess
        self._Augmenter = ImageAugmenter()
        self._imagesCap = imagesCap
        self._augmentationIter = augmentation_iter
        
        if(sum([usingAverage, usingMedian, usingMax]) > 1):
            print("You can't use more than one method to compare the embeddings, please choose only one.")
            exit(1)
        
        if(sum([usingAverage, usingMedian, usingMax]) == 0):
            self._usingMax = True
            print("Using max as default method to compare the embeddings.")
        
        if(weights != 'vggface2' and weights != 'casia-webface'):
            print(f"The weights '{weights}' are not avialable or do not exist.")
        
        #if(usingMtcnn):
        self._mtcnn = MTCNN(image_size=160, margin=0, select_largest=True, keep_all=False, selection_method='largest', post_process=True, device=self._device)
        self._model = InceptionResnetV1(pretrained=self._weights).eval()
        try:
            self._blacklistEmbeddings = torch.load(self._blacklistEmbeddingsFilename)
        except:
            print(f"Impossible to load pytorch embedding file, remember to build one before.\n Filename: '{self._blacklistEmbeddingsFilename}'")


    # building
    def buildBlacklistEmbeddings(self, blacklistFolderName="blacklist") -> None:
        self._blacklistFolderName = join(self._workspacePath, 'src', blacklistFolderName)
        
        self._blacklistEmbeddings = []
        
        blacklist_images = listdir(self._blacklistFolderName)
        blacklist_images = [self._blacklistFolderName + '/' + e for e in blacklist_images]

        process = 0
        
        for image_path in blacklist_images:
            currentImage = Image.open(image_path)
            croppedImage = self._mtcnn(currentImage)
            if(self._visualize):
                imshow("original image", currentImage)
                imshow("cropped image", croppedImage)
            if(croppedImage is None):
                continue
            
            with torch.no_grad():
                img_embeddings = []
                images_aug = []
                images_aug.append(croppedImage.unsqueeze(0)) # first image is the original one
                if(self._augmentationIter > 0):
                    for i in range(self._augmentationIter): # 1 img becomes len(transforms_list) images, this repeated for augmentationIter times
                        # N = 1 (curr_img) + len(transforms_list) * augmentationIter
                        self._augmenter.apply_transforms(croppedImage)
                        self._augmenter.convertToTensor()
                        l = self._augmenter.getTransformedTensors()
                        images_aug.extend(l)    # original + len(transforms_list) images
                        if(self._visualize):
                            imshow(f"augmented image {i+1}/{self._augmentationIter}", images_aug[-1])
                    for img in images_aug:
                        if(img.shape[0] != 3 and img.ndim == 3):
                            continue
                        if(img.ndim == 3):
                            img=img.unsqueeze(0)
                        img=img.float()
                        img_embeddings.append(self._model(img))
                else:
                    img_embeddings.append(self._model(croppedImage.unsqueeze(0)))
            self._blacklistEmbeddings.extend(img_embeddings)
            if(self._debug and self._augmentationIter == 0):
                print(f" --> {process} : {len(blacklist_images)}")
            if(self._debug and self._augmentationIter > 0):
                print(f" --> {process * (self._augmentationIter + 1)} : {len(blacklist_images)} * {self._augmentationIter + 1}")
            process += 1

        torch.save(self._blacklistEmbeddings, self._blacklistEmbeddingsFilename)
        if(self._augmentationIter > 0):
            self._areEmbeddingsAugmented = True
    
    
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
        self._distances.clear()
        #if(type(input_image) == np.ndarray):
        #    input_image = self.toPilImage(input_image)


        with torch.no_grad():
            if(self._usingMtcnn):   # 3*160*160
                if(self._visualize):
                    imshow('mtcnn in',input_image)
                    
                input_cropped = self._mtcnn(self.toPilImage(input_image))
                
                if(input_cropped is None):
                    return 3        # fallback
                inference_embedding = self._model(input_cropped.unsqueeze(0))
                if(self._visualize):
                    imshow("mtcnn out",input_cropped.permute(1, 2, 0).numpy())
                    if(self._mtcnnShowLandmarksPostProcessing):
                        boxes, probs, landmarks = self._mtcnn.detect(input_image, landmarks=True) # type: ignore
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
                resized = imresize(x, height=160)
                if(self._visualize):
                    imshow('resized',resized)

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
    
    
    
    def __computeAccOnList(self, imglist:list, isTP:bool):
        n_images = len(imglist)
        positive_list = []
        error_list = []
        self._distances = []
        res = {}
        cont = 0
        for img in imglist:
            cont+=1
            if(self._debug):
                print(f"Computing {cont}/{n_images} state: {'TP' if isTP else 'TN'}\n")

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

            for features in self._blacklistEmbeddings:
                dist = torch.Tensor([0])
                if(self._distanceMetric == 'cosine'):
                    cos = CosineSimilarity(dim=1, eps=1e-6)
                    dist = cos(features.squeeze(0), input_embedding)
                elif(self._distanceMetric == 'L2'):
                    dist = torch.cdist(features.squeeze(0), input_embedding.unsqueeze(0), 2) # type: ignore
                self._distances.append(dist.item())

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
                
                
            target='F'
            if distance >= self._distanceThreshold:
                positive_list.append(img)
                target='T'
            res[img] = target    

            input_image.close()
            
            if(self._imagesCap > 0 and cont == self._imagesCap):
                break
        return res
    
    
    def __computeMetrics(self, counter_TP:Counter, counter_TN:Counter):
        
        precision = counter_TP['T']/(counter_TP['T']+counter_TN['T'])
        recall = counter_TP['T']/(counter_TP['T']+counter_TP['N'])
        
        m = Metrics(precision, recall)      

        return m
    
    def __computeAccuracySide(self, testSetPath, truePositivePath) -> dict:
        tp_images, TP_number =  self.__get_image_files(truePositivePath)
        tn_images, TN_number = self.__get_image_files(testSetPath)
        res_TP = {}
        res_TN = {}
        
        res_TP['details'] = self.__computeAccOnList(tp_images, isTP=True)
        res_TN['details'] = self.__computeAccOnList(tn_images, isTP=False)
        
        counter_TP = Counter(res_TP['details'].values())
        counter_TN = Counter(res_TN['details'].values())
        # compute precision        
        
        metrics = self.__computeMetrics(counter_TP, counter_TN)
        
        
        
        res_TP['outcome_summary'] = {
                                    "total_true_positives_dataset_size": TP_number,
                                    "detected_positives": counter_TP['T'],
                                    "false_negatives": counter_TP['F'],
                                    "errors": counter_TP['E'],
                                    "accuracy": (counter_TP['T']/(TP_number - counter_TP['E']))*100 if self._imagesCap == 0 else (counter_TP['T']/self._imagesCap)*100,
                                    }
        
        res_TN['outcome_summary'] = {"total_true_negatives_dataset_size": TN_number,
                                     "detected_negatives": counter_TN['F'],
                                     "false_positives": counter_TN['T'],
                                     "errors": counter_TN['E'],
                                     "accuracy": (counter_TN['F']/(TN_number - counter_TN['E']))*100 if self._imagesCap == 0 else (counter_TN['F']/self._imagesCap)*100,
                                    }
        test_session_info = {
                            "using_image_cap": True if self._imagesCap > 0 else False,
                            "image_cap_value": self._imagesCap,
                            "threshold_used" : self._distanceThreshold,
                            "distance_metric_used": self._distanceMetric,
                            "pretrained_face_weights": self._weights,
                            "are_embeddings_augmented": self._areEmbeddingsAugmented,
                            "augmentation_iter": self._augmentationIter,
        }
        
        metrics = {
            "precision": metrics.getPrecision(),
            "recall": metrics.getRecall(),
            "f1_score": metrics.getF1Score() if metrics.isF1Computed() else "Not Computed",
        }
        
        res = {"metrics" : metrics, "test_session_info": test_session_info, "true_positives:" : res_TP, "true_negatives:" : res_TN }
        
        return res



    def computeAccuracy(self, testSetPath, truePositivePath, resultsFileName="test_results.json") -> None:
        res = self.__computeAccuracySide(testSetPath, truePositivePath)
        #counter = Counter(res['detected'].values())


        # print(f"Total: {n}")
        # print(f"Actual TP: {tp_n_images}")
        # print(f"target found: {counter['T']}\ntarget not found: {counter['F']}\ntarget error: {counter['E']}\n")

        # if self._debug:
        #     print("Positive targets:")
        #     for e in positive_list:
        #         print(e)
        #     print("Error targets:")
        #     for e in error_list:
        #         print(e)

        with open(resultsFileName, "w") as file:
            dump(res, file)
            
        # Crea il grafico di accuracy
        # labels = ['Target Found', 'Target Not Found', 'Target Error']
        # values = [counter['T'], counter['F'], counter['E']]

        # plt.figure()
        # plt.bar(labels, values)
        # plt.xlabel('Categories')
        # plt.ylabel('Counts')
        # plt.title(f'InceptionResnet: {self._weights} Metric: {self._distanceMetric}')
        # plt.savefig("Figura")


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
        
    def __getTextualOutputFromResult(self) -> str:
        retrieval_label = {1:'Detected and identified', 2:'Detected but not identified', 3: 'Not available yet',}
        return retrieval_label[self._status]
    
    def setUsingMtcnn(self, usingMtcnn:bool) -> None:
        self._usingMtcnn = usingMtcnn
        
    def setWeights(self, weights) -> None:
        if(self._weights != weights):
            print("Warning: weights got changed, model may not be consistent anymore for testing.")
        self._weights = weights