import cv2

image = cv2.imread(r'../images/6-1.png')
image_height, image_width, _ = image.shape
print(image_height, image_width)