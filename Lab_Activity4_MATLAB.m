% Read the image
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
figure(2); imshow(img_gray); title('Grayscale');

% Add blur to the image 
len = 21; theta = 11;
psf = fspecial('motion', len, theta);
img_blur = imfilter(img_gray, psf, 'conv', 'circular');

% Show the image 
figure(3); imshow(img_blur); title('Motion Blurred Image');

% Filtering Techniques

% Gaussian filtering
h_gaussian = fspecial('gaussian', [5, 5], 1); 
img_gaussian_filtered = imfilter(img_blur, h_gaussian);

% Display the Gaussian filtered image 
figure(4); imshow(img_gaussian_filtered); title('Filtered Image (Gaussian)');

% Sharpening using unsharp masking 
img_sharpened = imsharpen(img_blur);

% Display the sharpened image 
figure(5); imshow(img_sharpened); title('Sharpened Image');

% Add Gaussian noise and remove it using median filter 
img_noisy = imnoise(img_gray, 'gaussian', 0.02); 
img_noisy_removed = medfilt2(img_noisy, [5,5]);

% Display the noise image 
figure(6); imshow(img_noisy); title('Noisy');

% Display the noise-removed images 
figure(7); imshow(img_noisy_removed); title('Noise Removed');

% Deblurring 
estimated_nsr = 0.01;
img_deblurred = deconvwnr(img_blur, psf, estimated_nsr); 
figure(8); imshow(img_deblurred); title("Deblurred Image");

