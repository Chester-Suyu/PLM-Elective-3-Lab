pkg load image;
pkg load statistics;
% Global Image thresholding using Otsu's method
% load image
img = imread('flower.jpg');
figure(12);
imshow(img);
title('Acquire an Image of a Flower');
% calculate threshold using graythresh
level = graythresh(img);

% convert into binary image using the computed threshold
bw = im2bw(img,level);

% display the original image and the binary image

figure(1);
imshowpair(img, bw, 'montage');
title('Original Image (left) and Binary Image (right)');

% Multi-level thresholding using Otsu's method
img_gray = rgb2gray(img);

% Segment the image into two regions using the imquantize function, specifying the threshold level returned by the multithresh function.
seg_img = imquantize(img_gray,122);

% Display the original image and the segmented image
figure(2);
imshowpair(img,seg_img,'montage');
title('Original	Image	(left)	and Segmented Image (right)');

% Global histogram threshold using Otsu's method
% Calculate a 16-bin histogram for the image
[counts,x] = imhist(img,16);
figure(3);
stem(x,counts)

% Compute a global threshold using the histogram counts
T = otsuthresh(counts);

% Create a binary image using the computed threshold and display the image
bw2 = im2bw(img_gray,T);
figure(4);
imshow(bw2);
title('Binary Image');


%2. Region-based segmentation
% Using K means clustering
img2 = imread('flower.jpg');

% Convert the image to grayscale
bw_img2 = rgb2gray(img2);

% Segment the image into three regions using k-means clustering
[rows, cols, colors] = size(img2);
pixels = reshape(img, rows*cols, colors);
k = 3;
[idx, C] = kmeans(double(pixels), k);
segmented_img = zeros(rows*cols, 3);
for i = 1:k
    segmented_img(idx == i, :) = repmat(C(i, :), sum(idx == i), 1);
end
segmented_img = reshape(segmented_img, rows, cols, colors);
segmented_img = uint8(segmented_img);
figure(5);
imshow(segmented_img);
title('Labled Image');

% Using connected-component labeling, convert the image into binary
bin_img2 = im2bw(bw_img2);

% Label the connected components
[labeledImage, numberOfComponents] = bwlabel(bin_img2);

% Display the number of connected components
disp(['Number of connected components: ', num2str(numberOfComponents)]);

% Assign a different color to each connected component
coloredLabels = label2rgb(labeledImage, 'hsv', 'k', 'shuffle');

% Display the labeled image
figure(6);
imshow(coloredLabels);
title('Labeled Image');

% Paramter Modifications
% Adding noise to the image then segmenting it using otsu's method
img_noise = imnoise(img,'salt & pepper',0.09);
img_noise_gray = rgb2gray(img_noise);

% Calculate single threshold using multithresh
level = 124;

% Segment the image into two regions using the imquantize function, specifying the threshold level returned by the multithresh function.
seg_img = imquantize(img_noise_gray,level);

% Display the original image and the segmented image
figure(7);
imshowpair(img_noise,seg_img,'montage');
title('Original Image (left) and Segmented Image with noise (right)');

% Segment the image into two regions using k-means clustering
RGB = imread('flower.jpg');
[rows, cols, colors] = size(RGB);
pixels = reshape(img, rows*cols, colors);
k = 2;
[idx, C] = kmeans(double(pixels), k);
segmented_img = zeros(rows*cols, 3);
for i = 1:k
    segmented_img(idx == i, :) = repmat(C(i, :), sum(idx == i), 1);
end
segmented_img = reshape(segmented_img, rows, cols, colors);
segmented_img = uint8(segmented_img);
figure(8);
imshow(segmented_img);
title('Labeled Image');

%{
% Create a set of 24 Gabor filters, covering 6 wavelengths and 4 orientations
% Filter the grayscale image using the Gabor filters. Display the 24 filtered images in a montage
% Step 1: Define the wavelengths and orientations
wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
% Step 2: Create Gabor filters (Note: gabor function in Octave is different, so we'll use a loop)
figure(9);
montage(gabormag_montage, 'Size', [4 6]);

% Smooth each filtered image to remove local variations. Display the smoothed images in a montage
for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),3*sigma);
end
figure(10);
montage(gabormag,"Size",[4 6])

% Get the x and y coordinates of all pixels in the input image
nrows = size(RGB,1);
ncols = size(RGB,2);
[X,Y] = meshgrid(1:ncols,1:nrows);
% Convert gabormag (cell array) to a 3D array
num_filters = numel(gabormag);
gabormag_combined = zeros(nrows, ncols, num_filters);
for i = 1:num_filters
    gabormag_combined(:,:,i) = gabormag{i};
end
featureSet = cat(3, bw_RGB, gabormag_montage(:,:,1,:), X, Y);
featureSet = reshape(featureSet, [], size(featureSet, 3));
featureSet = zscore(featureSet);

% Segment the image into two regions using k-means clustering with the supplemented feature set
k = 2;
[idx, C] = kmeans(featureSet, k, 'MaxIter', 1000);
L2 = reshape(idx, nrows, ncols);
C = labeloverlay(RGB, L2);
figure(11);
imshow(C);
title("Labeled Image with Additional Pixel Information");
}%
