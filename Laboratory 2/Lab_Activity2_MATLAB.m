% Read an image
img = imread('flower1.jpg');

% Display the image 
figure(1);
imshow(img); 
title('Original Image');

% Get image dimensions (rows, columns, color channels) 
[rows, cols, channels] = size(img);
disp(['Image size: ', num2str(rows), ' x ', num2str(cols), ' x ', num2str(channels)]);

% Check color model (grayscale or RGB) 
if channels == 1
    disp('Color Model: Grayscale'); 
else
    disp('Color Model: RGB'); 
end

% Access individual pixels (example: center pixel) 
center_row = floor(rows/2) + 1;
center_col = floor(cols/2) + 1;
center_pixel = img(center_row, center_col, :); 
disp(['Center pixel value: ', num2str(center_pixel)]);

% Basic arithmetic operations (add constant value to all pixels) 
brightened_img = img + 50;
figure(2);
imshow(brightened_img); 
title ('Image Brightened');

% Basic geometric operation (flipping image horizontally) 
flipped_img = fliplr(img);
figure(3);
imshow(flipped_img); 
title('Image Flipped Horizontally');
 
% Gets the image channels
img_red = img;
img_red(:,:,2) = 0;
img_red(:,:,3) = 0;
subplot(131);
imshow(img_red);title("Red Channel");
img_green = img;
img_green(:,:,1) = 0;
img_green(:,:,3) = 0;
subplot(132);
imshow(img_green);title("Green Channel");
img_blue = img;
img_blue(:,:,1) = 0;
img_blue(:,:,2) = 0;
subplot(133);
imshow(img_blue);title("Blue Channel");

% Darken the image
darkened_img = img - 50;
subplot(121);
imshow(img);
title('Original Image');
subplot(122);
imshow(darkened_img);
title('Image Darkened');

% Darken and Brighten the image using Multiplication
brighten_img = img * 1.5;
darken_img = img * 0.5;
subplot(131);
imshow(img);
title('Original Image');
subplot(132);
imshow(brighten_img);
title('Image Brightened');
subplot(133);
imshow(darken_img);
title('Image Darkened');

% Rotates the image
img_rotate = imrotate(img,90);
subplot(121);
imshow(img);
title("Original Image");
subplot(122);
imshow(img_rotate);
title("Rotated Image");