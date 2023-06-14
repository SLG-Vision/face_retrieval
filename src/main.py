from retrieval import Retrieval
import cv2
from os.path import dirname, abspath, join

retr = Retrieval("def_blacklist.pt", threshold=0.18, debug=True, distanceMetric='cosine', usingAverage=True, usingMedian=False, usingMax=False, toVisualize=True, usingMtcnn=False)

# Example of usage across the Building phase, opencv inference RT phase and testing phase


def building():
    retr.buildBlacklistEmbeddings()
    

def inference():
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        cv2.imshow("original_frame", frame)
        r_code, r_string = retr.evalFrameTextual(frame)
        print(f"{r_code} --> {r_string}") 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def testing():
    retr.computeAccuracy( join(dirname(abspath(__file__)), 'datasets','lfw' ), join(dirname(abspath(__file__)), 'datasets','TP' ))

def main():
    # building()
    inference()
    # testing()


if __name__ == "__main__":
    main()