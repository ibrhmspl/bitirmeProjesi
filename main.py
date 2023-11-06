import cv2
import numpy as np

image1 = cv2.imread('img.png')

if image1 is None:
    print("İlk resim dosyası yüklenemedi.")
    exit()

image2 = cv2.imread('beyin1.png')

if image2 is None:
    print("İkinci resim dosyası yüklenemedi.")
    exit()

if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

difference = cv2.absdiff(image1, image2)

gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

threshold = 60
_, thresholded = cv2.threshold(gray_difference, threshold, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

highlighted_image = image1.copy()

for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Highlighted Image", highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
