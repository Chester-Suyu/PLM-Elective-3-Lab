import cv2
import matplotlib.pyplot as plt
# Acquire the image
img = cv2.imread('flower.jpg')

# Display the original image
cv2.imshow('Original Image', img)

# Convert to grayscale if the image is RGB
if img.shape[2] == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img

# Display the grayscale image
cv2.imshow('Grayscale Image', img_gray)

# Contrast enhancement using cv2.convertScaleAbs
img_contrast_enhanced = cv2.convertScaleAbs(img_gray, alpha=1.25, beta=0)

# Display the contrast-enhanced image
cv2.imshow('Contrast Enhanced Image', img_contrast_enhanced)

# Histogram equalization
img_histeq = cv2.equalizeHist(img_gray)

# Display the histogram equalized image
cv2.imshow('Equalized Image', img_histeq)

# Filtering using average
img_avg_filtered = cv2.blur(img_gray, (5, 5))

# Display the average filtered image
cv2.imshow('Filtered Image (Average)', img_avg_filtered)

# Filtering using median
img_median_filtered = cv2.medianBlur(img_avg_filtered, 5)

# Display the median filtered image
cv2.imshow('Filtered Image (Median)', img_median_filtered)

# Filtering using average filter but different values
img_avg_filtered2 = cv2.blur(img_gray, (10, 10))

# Display the median filtered image
cv2.imshow('Filtered Image (Using Average but Different Values)', img_avg_filtered2)

# Filtering using median filter but different values
img_median_filtered2 = cv2.medianBlur(img_gray, 1)

# Display the median filtered image
cv2.imshow('Experimented Filtered Image (Median)', img_median_filtered2)

# Display histograms for comparison------------------------------------------------------------------------------------

# Grayscale histogram
hist1 = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
plt.figure(1)
plt.title('Histogram of Grayscale')
plt.plot(hist1, color='black')

# Enhanced histogram
hist2 = cv2.calcHist([img_contrast_enhanced], [0], None, [256], [0, 256])
plt.figure(2)
plt.title('Histogram of Enhanced Image')
plt.plot(hist2, color='black')

# Equalized histogram
hist3 = cv2.calcHist([img_histeq], [0], None, [256], [0, 256])
plt.figure(3)
plt.title('Histogram of Equalized Image')
plt.plot(hist3, color='black')

# Histogram (Average Filtered)
hist4 = cv2.calcHist([img_avg_filtered], [0], None, [256], [0, 256])
plt.figure(4)
plt.title('Histogram of Average Filtered')
plt.plot(hist4, color='black')

# Histogram (Median Filtered)
hist5 = cv2.calcHist([img_median_filtered], [0], None, [256], [0, 256])
plt.figure(5)
plt.title('Histogram of Median Filtered')
plt.plot(hist5, color='black')

# Histogram (Experimented Median Filtered)
hist6 = cv2.calcHist([img_median_filtered2], [0], None, [256], [0, 256])
plt.figure(6)
plt.title('Histogram of Experimented Median Filtered')
plt.plot(hist6, color='black')

plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

