# Required imports
from collections import deque
import numpy as np
import cv2
import sys

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (550, 400))

# Parameters class include important paths and constants
class Parameters:
    def __init__(self,videopath):
        print(videopath)
        self.CLASSES = open("model/action_recognition_kinetics.txt"
                            ).read().strip().split("\n")
        self.ACTION_RESNET = 'model/resnet-34_kinetics (2).onnx'
        self.VIDEO_PATH = "test/"+videopath
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

if len(sys.argv) < 2:
    print("Error: Filename not provided.")
    sys.exit(1)

filename = sys.argv[1]

# Initialise instance of Class Parameter with the provided filename
param = Parameters(filename)

# A Double ended queue to store our frames captured and with time
# old frames will pop out of the deque
captures = deque(maxlen=param.SAMPLE_DURATION)

# load the human activity recognition model
#print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)

#print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)

while True:
    # Loop over and read capture from the given video input
    (grabbed, capture) = vs.read()

    # break when no frame is grabbed (or end if the video)
    if not grabbed:
        print("[INFO] no capture read from stream - exiting")
        break

    # resize frame and append it to our deque
    capture = cv2.resize(capture, dsize=(550, 400))
    captures.append(capture)

    # Process further only when the deque is filled
    if len(captures) < param.SAMPLE_DURATION:
        continue

    # now that our captures array is filled we can
    # construct our image blob
    # We will use SAMPLE_SIZE as height and width for
    # modifying the captured frame
    imageBlob = cv2.dnn.blobFromImages(captures, 1.0,
                                       (param.SAMPLE_SIZE,
                                        param.SAMPLE_SIZE),
                                       (114.7748, 107.7354, 99.4750),
                                       swapRB=True, crop=True)

    # Manipulate the image blob to make it fit as input
    # for the pre-trained OpenCV's Human Action Recognition Model
    imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
    imageBlob = np.expand_dims(imageBlob, axis=0)

    # Forward pass through model to make prediction
    net.setInput(imageBlob)
    outputs = net.forward()
    # Index the maximum probability
    label = param.CLASSES[np.argmax(outputs)]

    # Show the predicted activity 528491
    #cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
    #cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
     #           0.8, (0, 0, 0), 2)
    
    # Draw a rectangle around the region where activity is recognized
    cv2.rectangle(capture, (20, 50), (350, 350), (0, 255, 0), 2)
    
    # Display the label below the rectangle
    cv2.putText(capture, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0, 0, 255), 2)

    # Display it on the screen
    cv2.imshow("Human Activity Recognition", capture)
    key = cv2.waitKey(1) & 0xFF
    # Press key 'q' to break the loop
    if key == ord("q"):
        break

# Release the video capture object and close all windows
vs.release()
cv2.destroyAllWindows()
