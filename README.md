# Object Tracking Project


## Introduction
Object tracking is a fundamental computer vision task that involves locating and following moving objects in a sequence of frames from a video. It has numerous applications, such as video surveillance, robotics, sports analysis, augmented reality, and autonomous vehicles. This project presents two algorithms for object tracking: one for single object tracking and another for multiple object tracking.

## Single Object Tracking

The single object tracking algorithm uses feature matching and template matching methods to track a single object in a video. The script also calculates the average distance the object has moved and the time it takes to process each frame.

### Dependencies

- OpenCV
- NumPy
- SciPy

### Usage

Replace the empty string `''` in the following line with the path to the template image (the object you want to track) in both the `template_matching()` and `feature_matching()` functions:

```python
template = cv2.imread('', 0)
```

Replace the empty string `''` in the following line with the path to the video file you want to track the object in:

```python
video = cv2.VideoCapture("")
```

Run the script using Python.
The algorithm will display the tracked object in real-time and save the output as an output.mp4 file. It will also print the average distance the object has moved and the time it took to process each frame.

## Multiple Object Tracking

The multiple object tracking algorithm utilizes several tracking algorithms provided by OpenCV to simultaneously track multiple objects within a video. These include the BOOSTING Tracker, MIL Tracker, KCF Tracker, TLD Tracker, MEDIANFLOW Tracker, CSRT Tracker, and the MOSSE Tracker. The script calculates the average distance each object has moved and the time taken to process each frame.

### Dependencies

- OpenCV
- NumPy
- SciPy

### Usage

Replace the string "your video" in the following line with the path to the video file you want to track objects in:

```python
videoPath = "your video"
```

Run the script using Python. The algorithm will open a window named 'MultiTracker', where the first frame of the video is displayed. Draw bounding boxes around the objects you wish to track by clicking and dragging. Press any key to select the next object and press 'q' when you are finished selecting objects.

The algorithm will subsequently display the tracked objects in real-time, and it will also print the average distance each object has moved and the time it took to process each frame.

## Demo

A demo video of the single object tracking algorithm tracking an F-16: 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/36018286/129602591-d47b5caa-86f1-4dca-903d-c0d1f989acbb.gif)



