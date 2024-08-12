import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import color, filters
from skimage import measure
from scipy.ndimage import convolve
from skimage.filters import threshold_otsu, gabor_kernel, gabor
from skimage.util import random_noise
img = cv2.imread('flower.jpg')
cv2.imshow('Acquire an Image of a Flower', img)

# Global Image thresholding using Otsu's method
level = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
img_pair = cv2.hconcat([img,level])
cv2.imshow('Original Image (left) and Binary Image (right)',img_pair)

# Multi-level thresholding using Otsu's method
img2_PIL = Image.open('flower.jpg').convert('RGB')
img2_PIL = img2_PIL.quantize(2).convert('RGB')
seg_img = np.array(img2_PIL)[:,:,::-1].copy()
img_pair2 = cv2.hconcat([img,seg_img])
cv2.imshow('Original Image (left) and Segmented Image (right)', img_pair2)

# Create a binary image using the computed threshold and display the image
cv2.imshow('Binary Image', cv2.cvtColor(level, cv2.COLOR_BGR2GRAY))

# Region-Based Segmentation
# Using K-Means Clustering
img_Kmeans = img.reshape(-1,3)
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(img_Kmeans)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(img.shape)
plt.imshow(segmented_img/255)
plt.axis('off')
plt.title('Labled Image')
plt.show()

# Using connected-component labeling, convert the image into binary
img_gray = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)
thresh = threshold_otsu(img_gray)
bin_img2 = img_gray > thresh
labeledImage, numberOfComponents = measure.label(bin_img2,return_num=True)
print(f'Number of connected components: {numberOfComponents}')

# Paramter Modifications
# Adding noise to the image then segmenting it using otsu's method
img_noise = random_noise(img,mode='s&p', amount=0.09)
img_noise = (img_noise * 255).astype(np.uint8)
img_noise_gray = cv2.cvtColor(img_noise, cv2.COLOR_BGR2GRAY)
level = threshold_otsu(img_noise_gray)
seg_img = (img_noise_gray > level).astype(np.uint8) * 255
seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
img_pair = np.hstack([img_noise, seg_img])
cv2.imshow('Original Image (left) and Segmented Image with noise (right)',img_pair)

#Segment the image into two regions using k-means clustering
img_Kmeans = seg_img.reshape(-1,3)
kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(img_Kmeans)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(seg_img.shape)
plt.imshow(segmented_img/255)
plt.axis('off')
plt.title('Labeled Image')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()