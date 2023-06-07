from retrieval import Retrieval
import cv2


def main():
    
    retr = Retrieval('m_blacklist.pt')
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        r = retr.evaluate(frame)
        if(r):
            print("Found")
        else:
            print("Not found")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()