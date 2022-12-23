import cv2
from toolkit import Capturer, Predictor

kf = Predictor()

counter = 0
differ = 0
title = 'Predetection'

while True:

    # 模拟来回移动
    if counter >= 10:
        differ = -1
    elif counter <= 1:
        differ = 1
    counter += differ

    img = Capturer.grab(region=(0, 0, 1000, 200), convert=True)

    # 在图片上画来回移动
    point = (counter * 50 + 200, 100)
    cv2.circle(img, point, 20, (0, 20, 220), -1)

    # 在图片上画预测的移动位置
    predicted = kf.predict(point)
    cv2.circle(img, predicted, 21, (20, 220, 0), 2)

    print(predicted[0] - point[0])

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, img)
    k = cv2.waitKey(500)
    if k % 256 == 27:
        cv2.destroyAllWindows()
        exit('ESC ...')











