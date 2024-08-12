import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sc
from skimage import restoration,io
#Read the image
img = cv2.imread('flower.jpg')
#Display the original image
#cv2.imshow('Original Image', img)
#Convert to graysacle if the image is RGB
if img.shape[2] == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img
#Display the grayscale image
#cv2.imshow('Grayscale', img_gray)
#Add blur to the image
kernel_motion_blur = np.zeros((21, 21))
kernel_motion_blur[10, :] = np.ones(21)
kernel_motion_blur = kernel_motion_blur / 21
img_blur = cv2.filter2D(img_gray, -1, kernel_motion_blur)
#Show the image
#cv2.imshow('Motion Blurred Image', img_blur)
#Gaussian filtering
img_gaussian_filtered = sc.gaussian_filter(img_blur, sigma=3)
#Display the Gaussian filtered image
#cv2.imshow('Filtered Image (Gaussian)', img_gaussian_filtered)
#Sharpening using unsharp masking
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
img_sharpened = cv2.filter2D(img_blur, -1, kernel)
#Display the sharpened image
#cv2.imshow('Sharpened Image', img_sharpened)
#Add Gaussian noise and remove it using median filter
noise = np.random.normal(25, 15, img_gray.shape).astype(np.uint8)
img_noisy = cv2.add(img_gray, noise)
img_noisy_removed = cv2.medianBlur(img_noisy, 5)
#Display the noise image
#cv2.imshow('Noisy', img_noisy)
#Display the noise-removed images
#cv2.imshow('Noise Removed', img_noisy_removed)
#Deblurring
img2 = io.imread('flower.jpg', as_gray=True)
kernel_motion_blur = np.zeros((21, 21))
kernel_motion_blur[10, :] = np.ones(21)
kernel_motion_blur = kernel_motion_blur / 21
img_blur2 = cv2.filter2D(img2, -1, kernel_motion_blur)
img_deblurred = restoration.wiener(img_blur2,np.ones((1,1)),1)
#cv2.imshow('Deblurred Image', img_deblurred)
#Parameter Modification
#Gaussian Filtering
img_gaussian_filtered2 = sc.gaussian_filter(img_blur, sigma=5)
cv2.imshow('Filtered Image with Experimented Value (Gaussian)', img_gaussian_filtered2)
#Histogram (Gaussian Filtered)
hist1 = cv2.calcHist([img_gaussian_filtered2], [0], None, [256], [0, 256])
plt.figure(1)
plt.plot(hist1)
plt.title('Histogram of the Experimented Value (Gaussian Filtered)')
#Add Gaussian noise
noise1 = np.random.normal(100, 15, img_gray.shape).astype(np.uint8)
noise2 = np.random.normal(25, 15, img_gray.shape).astype(np.uint8)
img_noisy_exp1 = cv2.add(img_gray, noise1)
img_noisy_exp2 = cv2.add(img_gray, noise2)
#Display the noisy
cv2.imshow('Noisy Using Experimented Value (Gaussian is 0.5)', img_noisy_exp1)
cv2.imshow('Noisy Using Experimented Value (Gaussian is 0.1)', img_noisy_exp2)
#Display the histogram for Noisy
hist2 = cv2.calcHist([img_noisy_exp1], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([img_noisy_exp2], [0], None, [256], [0, 256])
plt.figure(2)
plt.plot(hist2)
plt.title('Histogram of Noisy Image Experimented Value 1')
plt.figure(3)
plt.plot(hist3)
plt.title('Histogram of Noisy Image Experimented Value 2')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
