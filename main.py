from utilis import *
import pytesseract
path = 'C:\\Users\\Aryan Gupta\\OneDrive\\Pictures\\1.jpg'
hsv = [0, 65, 59, 255, 0, 255]
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread(path)
imgResult = detectColor(img, hsv)
imgContours, contours = getContours(imgResult, img, showCanny=True,
                                    minArea=1000, filter=4,
                                    cThr=[100, 150], draw=True)
cv2.imshow("imgContours", imgContours)
print(len(contours))
roiList = getRoi(img, contours)
roiDisplay(roiList)
highlightedText = []
for x, roi in enumerate(roiList):
    highlightedText.append(pytesseract.image_to_string(roi))
saveText(highlightedText)
imgStack = stackImages(0.7, ([img, imgResult, imgContours]))
cv2.imshow("Stacked Images", imgStack)
cv2.waitKey(0)