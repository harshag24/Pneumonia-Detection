########################################################################################
### Closing and Opening
import cv2
import numpy as np

# Reading image from its path
img = cv2.imread(".\\chest_xray\\chest_xray\\train\\NORMAL\\NORMAL-28501-0001.jpeg")
# img = cv2.imread(".\\chest_xray\\chest_xray\\train\\PNEUMONIA\\BACTERIA-7422-0001.jpeg")

# Resizing image to (256 x 256)
img = cv2.resize(img, (256, 256))

# Defining Kernel
kernel = np.ones((5, 5), np.uint8)

# Using closing and opening
img = cv2.erode(img, kernel)
img = cv2.dilate(img, kernel)
img = cv2.erode(img, kernel)

# Displaying the image
cv2.imshow("Image", img)
cv2.waitKey(0)

########################################################################################
### High Pass Filter
import cv2
import numpy as np

# Reading image from its path
img = cv2.imread(".\\chest_xray\\chest_xray\\train\\NORMAL\\NORMAL-28501-0001.jpeg")

# Resizing image to (256 x 256)
img = cv2.resize(img, (256, 256))

# Using a high pass filter
kernel = np.array([[-1.0, -1.0, -1.0],
                   [-1.0, 8.0, -1.0],
                   [-1.0, -1.0, -1.0]])


kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

img = cv2.filter2D(img, -1, kernel)

# Displaying the image
cv2.imshow("Image", img)
cv2.waitKey(0)

########################################################################################
### Low Pass Filter
import cv2
import numpy as np

# Reading image from its path
img = cv2.imread(".\\chest_xray\\chest_xray\\train\\NORMAL\\NORMAL-28501-0001.jpeg")
# img = cv2.imread(".\\chest_xray\\chest_xray\\train\\PNEUMONIA\\BACTERIA-7422-0001.jpeg")

# Resizing image to (256 x 256)
img = cv2.resize(img, (256, 256))

# Using a low pass filter
kernel = (1.0 / 9.0) * np.array([[1.0, 1.0, 1.0],
                                 [1.0, 1.0, 1.0],
                                 [1.0, 1.0, 1.0]])

print(kernel)

kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)

img = cv2.filter2D(img, -1, kernel)

# Displaying the image
cv2.imshow("Image", img)
cv2.waitKey(0)

########################################################################################
### Sobel Operator
import cv2
import numpy as np

# Reading image from its path
# img = cv2.imread(".\\chest_xray\\chest_xray\\train\\NORMAL\\NORMAL-28501-0001.jpeg")
img = cv2.imread(".\\chest_xray\\chest_xray\\train\\PNEUMONIA\\BACTERIA-7422-0001.jpeg")

# Resizing image to (256 x 256)
img = cv2.resize(img, (256, 256))

# Using Sobel operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

abs_grad_x = cv2.convertScaleAbs(sobelx)
abs_grad_y = cv2.convertScaleAbs(sobely)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Displaying the image
cv2.imshow("Image", grad)
cv2.waitKey(0)

########################################################################################
### Thresholding with matlab
%%% Reading input image
img=imread('Images\8.jpeg');
% img=rgb2gray(img);
img=imresize(img,[256,256],'nearest');

temp = zeros([256, 256]);

% J = histeq(img);

for i = 1:256
    for j = 1:256
        if img(i, j) > 150
            temp(i, j) = 255;
        end
    end
end

figure(1);
subplot(1,2,1), imshow(img), title("Original");
subplot(1,2,2), imshow(temp), title("Thresholded image");
% subplot(2,2,3), imhist(img, 256), title("Original Histogram");
% subplot(2,2,4), imhist(temp, 256), title("Final Histogram");

% temp1 = zeros([256, 256]);
% 
% for i = 1:256
%     for j = 1:256
%         if J(i, j) > 150
%             temp1(i, j) = 255;
%         elseif J(i, j) > 120
%                 temp1(i, j) = 50;
%         end
%     end
% end
% 
% figure(2);
% subplot(2,2,1), imshow(J), title("Hist Eq img");
% subplot(2,2,2), imshow(temp1), title("Hist Eq Thresholded");
% subplot(2,2,3), imhist(J), title("Hist Eq Histogram");
% subplot(2,2,4), imhist(temp1), title("Hist Eq Thresholded Histogram");
