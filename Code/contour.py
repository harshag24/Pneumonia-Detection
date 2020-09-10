import cv2

img = cv2.imread('s.jpg')
img = cv2.resize(img, (604, 334))
cv2.imshow('original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.GaussianBlur(thresh, (9, 9), 0)

cont, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, cont, -1, (0, 255, 0), 2)
#img = cv2.resize(img , (604,334))
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
