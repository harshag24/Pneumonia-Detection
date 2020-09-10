import cv2
import numpy as np

img = cv2.imread('new.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 15, -15)
# Show binary image
# cv2.imshow("binary", bw)

horizontal = np.copy(bw)
# Specify size on horizontal axis
cols = horizontal.shape[1]
horizontal_size = cols // 30
# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
#Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)
#Show extracted horizontal lines
print(horizontal)
# cv2.imshow("horizontal", horizontal)

h = img.shape[0]
w = img.shape[1]
max_height=0
for y in range(0, h):
    for x in range(0, w):
        if horizontal[y][x]>0:
            max_height=y
            break

print(max_height)

cropped = img[max_height-10: , 0:w]
# cv2.imshow('cropped',cropped)


hsv_img = cv2.cvtColor(cropped , cv2.COLOR_BGR2HSV)

blue_lower = np.array([94,80,2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)

white_lower = np.array([0,0,250], np.uint8)
white_upper = np.array([255,255, 255], np.uint8)


white_mask = cv2.inRange(hsv_img, white_lower, white_upper)
cv2.imshow('White Mask' , white_mask)


kernal_white = np.ones((5, 5), "uint8")

white_mask = cv2.dilate(white_mask, kernal_white)
res_white = cv2.bitwise_and(cropped, cropped,mask = white_mask)

contours, hierarchy = cv2.findContours(white_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if (area > 350):
        x, y, w, h = cv2.boundingRect(contour)
        cropped = cv2.rectangle(cropped, (x, y),
                                   (x + w, y + h),
                                   (0,255, 0), 2)

        cv2.putText(cropped, "White Team", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0))


hsv_img = cv2.cvtColor(cropped , cv2.COLOR_BGR2HSV)

blue_mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
cv2.imshow('Blue Mask' , blue_mask)

kernal_blue = np.ones((5, 5), "uint8")

blue_mask = cv2.dilate(blue_mask, kernal_blue)
res_blue = cv2.bitwise_and(cropped, cropped,mask = blue_mask)

contours, hierarchy = cv2.findContours(blue_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if (area > 250):
        x, y, w, h = cv2.boundingRect(contour)
        cropped = cv2.rectangle(cropped, (x, y),
                                   (x + w, y + h),
                                   (255, 0, 0), 2)

        cv2.putText(cropped, "Blue Team", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0))

cv2.imshow("Team Detection", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()