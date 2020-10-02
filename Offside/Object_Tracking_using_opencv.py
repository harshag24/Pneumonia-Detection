import cv2

cap = cv2.VideoCapture(0)

# Install package "opencv-contrib-python" for trackers
#tracker = cv2.TrackerMOSSE_create()
tracker = cv2.TrackerCSRT_create()

success, img = cap.read()
bbox = cv2.selectROI("Tracker", img, False)
tracker.init(img, bbox)

def drawbox(img, bbox):
    x, y, width, height = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x+width), (y+height)), (0, 255, 0), 2, 2)

while True:
    success, img = cap.read()

    success, bbox = tracker.update(img)

    if(success):
        drawbox(img, bbox)
    else:
        cv2.putText(img, "Lost", (75, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(img, "Camera", (75, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


