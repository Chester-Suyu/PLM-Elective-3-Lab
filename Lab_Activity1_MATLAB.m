% Read the image
img = imread('flower.jpg');
% Rotate by 45 degrees
rotated_img = imrotate(img, 30);
% Flip horizontally
flipped_img = fliplr(rotated_img);
% Display results
figure(1);
plot(1,1);
imshow(img);
title('Original Image');
figure(2);
plot(1,1);
imshow(rotated_img);
title('Rotated 30°');
figure(3);
plot(1,1);
imshow(flipped_img);
title('Rotated & Flipped');