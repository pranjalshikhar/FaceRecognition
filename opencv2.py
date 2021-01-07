# Simple Progam to read and show an image

import cv2

img = cv2.imread("dog.png")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# NOTE : 
# While plotting the image with matplotlib, the image
# automatically converted into BGR. 
# And with the use of "imshow()" it remains to RGB only.


newimg = cv2.imread("dog.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Gray Image", newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()