from __future__ import print_function
import sys
import cv2
from random import randint
from scipy.spatial import distance as dist
import numpy as np
import time

#https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
sum_time = 0
count = 0

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

#Step 2: Read First Frame of a Video
# Set video to load
videoPath = "Video/messi.mp4"

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

#Step 3: Locate Objects in the First Frame
## Select boxes
bboxes = []
colors = []
centerCoord = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
  bbox = cv2.selectROI('MultiTracker', frame)
  bboxes.append(bbox)
  colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")
  k = cv2.waitKey(0) & 0xFF
  if (k == 113):  # q is pressed
    break

# Calculate centroid for each object
for (i, box) in enumerate(bboxes):
    centerCoord.append((box[0] + (box[2] / 2), box[1] + (box[3] / 2)))

print('Selected bounding boxes {}'.format(bboxes))

# Step 4: Initialize the MultiTracker
# Specify the tracker type
trackerType = trackerTypes[0]
#trackerType ='KCF'

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)

Distance = [0 for i in range(0, len(centerCoord))]

# Step 5: Update MultiTracker & Display Results
# Process video and track objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Start timer
    start_time = time.time()

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # Calculate the time that has passes
    end_time = time.time() - start_time
    sum_time = sum_time + end_time

    count = count + 1

    ncenterCoord = []

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        # Centroid calculation for each object
        ncenterCoord.append((newbox[0] + (newbox[2] / 2), newbox[1] + (newbox[3] / 2)))
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # Movement centroid calculation between previous and present frame for each object
    for i in range(0, len(centerCoord)):
        Distance[i] = Distance[i] + dist.cdist(np.array([centerCoord[i]]), np.array([ncenterCoord[i]]), 'euclidean')

    centerCoord = ncenterCoord

    # show frame
    cv2.imshow('MultiTracker', frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

for i in range(0, len(Distance)):
    result = Distance[i] / count
    print("Distance average of bounding box", i + 1, ": {:.2f}".format(result[0][0]))
print("Average of time that algorithm needs to process per frame: {:.4f}".format(sum_time/count))
print("Frame Number:", count)

cap.release()
cv2.destroyAllWindows()
