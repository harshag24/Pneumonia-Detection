import cv2
import numpy as np

img = cv2.imread('new.jpg')
#img = cv2.resize(img , (img.shape[1]//2, img.shape[0]//2))

# kernel = np.array([[-1,-1,-1],
#                    [-1,9,-1],
#                    [-1,-1,-1]])
#
# sharpened = cv2.filter2D(img , -1 , kernel)
# cv2.imshow('sharpened',sharpened)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#_,thresh = cv2.threshold(gray , 95 , 255 , cv2.THRESH_BINARY_INV)
#ret , thresh = cv2.threshold(gray , 0 , 255 , cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
thresh = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.Canny(thresh, 50, 150, apertureSize=3)
cv2.imshow('Thresh', thresh)

minLineLength = 80
maxLineGap = 1

lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength, maxLineGap)

# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
a, b, c = lines.shape
for i in range(a):
    cv2.line(thresh, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 0), 17)
cv2.imshow('houghlines5.jpg', thresh)

# fg = cv2.bitwise_or(img, img, mask=thresh)
#
# mask = cv2.bitwise_not(thresh)
#
# background = np.full(img.shape, 255, dtype=np.uint8)
#
# bk = cv2.bitwise_or(background, background, mask=mask)
#
# # combine foreground+background
# final = cv2.bitwise_or(fg, bk)

result = img.copy()
result[thresh != 0] = (120, 255, 0)

cv2.imshow('Final', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
