import cv2
import numpy as np

if __name__ == "__main__":
    image = cv2.imread("caps-images/3.jpg")
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=100,
        minRadius=0,
        maxRadius=0,
    )
    circles = np.round(circles[0, :]).astype("int")
    for x, y, r in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
