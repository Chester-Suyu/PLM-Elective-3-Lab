import cv2
#Read the image
img = cv2.imread(r"C:\Users\Chad\OctaveFiles\flower.jpg")
#Rotate by 45 degrees
center_img = (img.shape[1]//2, img.shape[0]//2)
rotation_img = cv2.getRotationMatrix2D(center_img, 30, 1)
rotated_img = cv2.warpAffine(img, rotation_img,(img.shape[1],img.shape[0]))
#Flip horizontally
flipped_img = cv2.flip(rotated_img,1)
#Display results
cv2.imshow('Original Image', img)
cv2.imshow('Rotated 30°', rotated_img)
cv2.imshow('Rotated & Flipped', flipped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
