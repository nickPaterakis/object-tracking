import cv2
import sys
import numpy as np
from scipy.spatial import distance as dist
import time


def template_matching(img_rgb) :

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # load the template image we look for
    # give an image of the object that you want to track. The image should be from the first frame
    # of the video
    template = cv2.imread('', 0)
    w, h = template.shape[::-1]

    # run the templae matching
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.99
    loc = np.where(res >= threshold)

    # mark the corresponding location(s)
    return (loc[1][0], loc[0][0], w, h)


def feature_matching(frame):
    MIN_MATCH_COUNT = 10

    # give an image of the object that you want to track. The image should be from the first frame
    # of the video
    img1 = cv2.imread('', 0)  # queryImage as grayscale

    # Initiate SIFT detector
    # sift  = cv2.ORB_create() # before opencv 3
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = np.int32(cv2.perspectiveTransform(pts, M))
        return (dst[1, 0, 0], dst[0, 0, 1], dst[2, 0, 0] - dst[1, 0, 0], dst[1, 0, 1] - dst[0, 0, 1])


if __name__ == '__main__':

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # Set up tracker.
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[0]
    sum_time = 0
    Distance = 0
    count = 0

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture("")
    centerCoord = []

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # save the first frame
    # cv2.imwrite('first_frame.jpg', frame)

    # bbox = template_matching(frame)
    bbox = feature_matching(frame)

    # Calculate the center of bounding box
    centerCoord = (bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2))

    # Start timer
    start_time = time.time()

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    # Calculate the time that has passes
    end_time = time.time() - start_time
    sum_time = sum_time + end_time

    # define fps, frame_width and frame height
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1280, 720))

    while True:
        ncenterCoord = 0

        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        start_time = time.time()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate the time that has passes
        end_time = time.time() - start_time
        sum_time = sum_time + end_time

        # Draw bounding box
        if ok:
            # Tracking success
            # Calculate the center of bounding box
            ncenterCoord = (bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Calculate the distance between bounding boxes
        Distance = Distance + dist.cdist(np.array([centerCoord]), np.array([ncenterCoord]), 'euclidean')

        centerCoord = ncenterCoord
        count = count + 1

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        out.write(frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    out.release()

    print("Distance average: {:.2f}".format(Distance[0][0] / count))
    print("Time: {:.2f}".format(sum_time))