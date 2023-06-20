from retrieval import Retrieval
import cv2
from os.path import dirname, abspath, join

retr = Retrieval("def_blacklist.pt", threshold=0.18, debug=True, distanceMetric='cosine', imagesCap=50, usingAverage=True, usingMedian=False, usingMax=False, toVisualize=True, usingMtcnn=True)

# Example of usage across the Building phase, opencv inference RT phase and testing phase


def build():
    retr.buildBlacklistEmbeddings()
    

def infer():
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        cv2.imshow("original_frame", frame)
        r_code, r_string = retr.evalFrameTextual(frame)
            
        if(retr.hasMtcnnFailed()):
            continue
        
        print(f"{r_code} --> {r_string}") 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def test():
    retr.computeAccuracy( join(dirname(abspath(__file__)), 'datasets','lfw' ), join(dirname(abspath(__file__)), 'datasets','TP_Test' ))

def main():
    # build()
    # infer()
    test()


if __name__ == "__main__":
    main()