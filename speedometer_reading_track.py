import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression_fast(contours, overlapThresh):
    boxes = []
    for cnt in contours:
        box = cv.boundingRect(cnt)
        boxes.append(box)
    boxes = np.array(boxes)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x = boxes[:,0]
    y = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    picked_contours = []
    for ind in pick:
        picked_contours.append(contours[ind])
    return picked_contours

cap = cv.VideoCapture("eighthday/speedometer.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
angles = []
while cap.isOpened():
    # Take each frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of some color in HSV
    color = np.uint8([[[58,21,121]]])
    hsv_color = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    # print(hsv_color)
    # Threshold the HSV image to get only red colors
    mask1 = cv.inRange(hsv, np.array([0,70,50]), np.array([10,255,255]))
    mask2 = cv.inRange(hsv, np.array([168,70,50]), np.array([180,255,255]))
    mask = mask1 | mask2

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    hull_contours = []
    for cnt in contours:
        hull_cnt = cv.convexHull(cnt)
        hull_contours.append(hull_cnt)

    hull_contours = non_max_suppression_fast(hull_contours, 0.2)

    res = frame.copy()

    hull_contours = sorted(hull_contours, key=lambda x: cv.arcLength(x, closed=True), reverse=True)
    for i in range(len(hull_contours)):
        cnt = hull_contours[i]
        if len(cnt) >= 5:
            ellipse = cv.fitEllipse(cnt)
            angles.append(ellipse[-1])
            cv.ellipse(res, ellipse, (255,255,255), 3)
            break

    cv.imshow('res', res)
    cv.imshow('mask', mask)
    k = cv.waitKey(int(1000/fps)) & 0xFF
    if k == ord('q'):
        break
cv.destroyAllWindows()

max_angle = max(angles)
min_angle = min(angles)

cap.set(cv.CAP_PROP_POS_FRAMES, angles.index(max_angle))
ret, frame_max = cap.read()
cap.set(cv.CAP_PROP_POS_FRAMES, angles.index(min_angle))
ret, frame_min = cap.read()

cv.imshow('max', frame_max)
cv.imshow('min', frame_min)
cv.waitKey(1)
cv.destroyAllWindows()

# YOU MUST MANUALLY INPUT THE LOWEST AND HIGHEST SPEED THE SPEEDOMETER HITS IN THE INPUT VIDEO HERE
rpm_points = [0.5, 2.6]
angle_points = [min_angle, max_angle]
rpm_values = np.interp(angles,  angle_points, rpm_points)

frames = [0, len(angles)]
seconds = [0, len(angles)//fps]
time = np.interp(range(len(angles)), frames, seconds)

fig = plt.figure(figsize=(6,cap.get(cv.CAP_PROP_FRAME_HEIGHT)/100))
plt.plot(time, rpm_values, color='gray')
plot, = plt.plot(time, rpm_values, color='red')
scat = plt.scatter(None, None, color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.tight_layout(pad=2)
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
video = []
for i in range(len(angles)):
    ret, frame = cap.read()
    # update data
    plot.set_ydata([None if j > i else val for j, val in enumerate(rpm_values)])
    scat.set_offsets((time[i], rpm_values[i]))

    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = cv.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv.COLOR_RGBA2BGR)
    img = cv.resize(img, (fig.canvas.get_width_height()[0], frame.shape[0]))

    # display image with opencv or any operation you like
    res = np.hstack((frame, img))
    video.append(res)
    cv.imshow("final", res)

    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cv.destroyAllWindows()

vw = cv.VideoWriter('eighthday/speedometer_reading.mp4', -1, fps, (res.shape[1],res.shape[0]))
for i in range(len(video)):
    vw.write(video[i])