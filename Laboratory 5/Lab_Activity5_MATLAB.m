% Global Image thresholding using Otsu's method
% load image
img = imread('flower.jpg');

% calculate threshold using graythresh
level = graythresh(img);

% convert into binary image using the computed threshold 
bw = imbinarize(img, level);

% display the original image and the binary image

figure(1); 
imshowpair(img, bw, 'montage'); 
title('Original Image (left) and Binary Image (right)');

% Multi-level thresholding using Otsu's method
% calculate single threshold using multithresh
img_gray = rgb2gray(img);
level2 = multithresh(img_gray);

% Segment the image into two regions using the imquantize function, specifying the threshold level returned by the multithresh function.
seg_img = imquantize(img_gray,level2);

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
bw2 = imbinarize(img_gray,T);
figure(4); 
imshow(bw2); 
title('Binary Image');


%2. Region-based segmentation
% Using K means clustering
img2 = imread('flower.jpg');

% Convert the image to grayscale
bw_img2 = rgb2gray(img2);

% Segment the image into three regions using k-means clustering
[L, centers] = imsegkmeans(bw_img2,3);
B = labeloverlay(bw_img2,L);
figure(5);
imshow(B); 
title('Labled Image');

% Using connected-component labeling, convert the image into binary
bin_img2 = imbinarize(bw_img2);

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
level = multithresh(img_noise_gray);

% Segment the image into two regions using the imquantize function, specifying the threshold level returned by the multithresh function.
seg_img = imquantize(img_noise_gray,level);

% Display the original image and the segmented image
figure(7); 
imshowpair(img_noise,seg_img,'montage'); 
title('Original Image (left) and Segmented Image with noise (right)');

% Segment the image into two regions using k-means clustering 
RGB = imread('flower.jpg');
L = imsegkmeans(RGB,2); 
B = labeloverlay(RGB,L);
figure(8); 
imshow(B); 
title('Labeled Image');

% Create a set of 24 Gabor filters, covering 6 wavelengths and 4 orientations 
wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength,orientation);

% Convert the image to grayscale 
bw_RGB = im2gray(im2single(RGB));

% Filter the grayscale image using the Gabor filters. Display the 24 filtered images in a montage
gabormag = imgaborfilt(bw_RGB,g); 
figure(9); 
montage(gabormag,"Size",[4 6])

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
[X,Y] = meshgrid(1:ncols,1:nrows); featureSet = cat(3,bw_RGB,gabormag,X,Y);

% Segment the image into two regions using k-means clustering with the supplemented feature set
L2 = imsegkmeans(featureSet,2,"NormalizeInput",true); C = labeloverlay(RGB,L2);
figure(11);
imshow(C); 
title("Labeled Image with Additional Pixel Information");
