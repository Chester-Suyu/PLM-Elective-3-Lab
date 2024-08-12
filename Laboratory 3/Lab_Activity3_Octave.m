pkg load image
% Read an image
img = imread('flower.jpg');

% Display the original image
figure(1); imshow(img); title('Original Image');

% Convert to grayscale if the image is RGB
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Display the grayscale image
figure(2); imshow(img_gray); title('Grayscale Image');

% Contrast enhancement using imadjust
img_contrast_enhanced = imadjust(img_gray);

% Display the contrast-enhanced image
figure(3); imshow(img_contrast_enhanced); title('Contrast Enhanced Image (imadjust)');

% Histogram equalization
img_histeq = histeq(img_gray);

% Display the histogram equalized image
figure(4); imshow(img_histeq); title('Equalized Image');

% Filtering using average filterh
h_avg = fspecial('average', [5, 5]); img_avg_filtered = imfilter(img_gray, h_avg);

% Display the average filtered image
figure(5); imshow(img_avg_filtered); title('Filtered Image (Average)');

% Filtering using median filter
img_median_filtered = medfilt2(img_gray, [5, 5]);

% Display the median filtered image
figure(6); imshow(img_median_filtered); title('Filtered Image (Median)');

% Display histograms for comparison

% Grayscale histogram
figure(7); imhist(img_gray); title('Histogram of Grayscale');

% Enhanced histogram (imadjust)
figure(8); imhist(img_contrast_enhanced); title('Histogram of Enhanced Image');

% Equalized histogram
figure(9); imhist(img_histeq); title('Histogram of Equalized Image');

% Histogram (Average Filtered)
figure(10); imhist(img_avg_filtered); title('Histogram of Average Filtered)');

% Histogram (Median Filtered)
figure(11); imhist(img_median_filtered); title('Histogram of Median Filtered)');

% Filtering using average filter but different values
h_avg = fspecial('average', [10, 10]); % Original is [5,5]
img_avg_filtered2 = imfilter(img_gray, h_avg);

% Show the experimented image
figure(12); imshow(img_avg_filtered2); title('Filtered Image (Using Average but Different values)');

% Filtering using median filter
img_median_filtered2 = medfilt2(img_gray, [1, 10]); % Original is [5,5]

% Display the median filtered image
figure(13); imshow(img_median_filtered2); title('Experimented Filtered Image (Median)');

% Show the Histogram
figure(14); imhist(img_median_filtered2); title('Histogram of Experimented Median Filtered)');
