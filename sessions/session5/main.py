import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# region function
def show_before_after(img_before, img_after, title_1: str = "Before", title_2: str = "After"):
    """
    نمایش تصاویر قبل و بعد از اعمال تغییرات و افکت با OpenCV
    :param img_before: تصویر اصلی
    :param img_after: تصویر بعد از اعمال تغییرات
    :param title_1:  عنوان سابپلات تصویر اصلی
    :param title_2: عنوان سابپلات تصویر بعد از اعمال تغییرات
    """
    # تبدیل BGR به RGB
    if len(img_before.shape) == 3:
        img_before_rgb = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        img_after_rgb = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
    else:
        img_before_rgb = img_before
        img_after_rgb = img_after

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img_before_rgb)
    ax1.set_title(title_1)
    ax1.axis('off')

    ax2.imshow(img_after_rgb)
    ax2.set_title(title_2)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


# endregion


#region Part 1 -- Put Text
# img = cv2.imread("images/cute-cats.bmp")
# rows, cols, c = img.shape
#
# pt_1 = (100, 280)
# pt_2 = (370, rows-1)
# color = (128/255, 64/255, 32/255)
#
# outlier_color = (182/255, 105/255, 116/2555)
# cv2.rectangle(img, pt_1, pt_2, outlier_color, 2)
#
# text = "Cute Cats"
# font = cv2.putText(img, text, (100,275), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=outlier_color, thickness=2, lineType=cv2.LINE_AA)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("images/cute-cats.bmp")
# rows, cols, c = img.shape
#
# pt_1 = (450, 70)
# pt_2 = (600, 170)
# color = (128/255, 64/255, 32/255)
#
# outlier_color = (182/255, 105/255, 116/2555)
# cv2.rectangle(img, pt_1, pt_2, outlier_color, 2)
#
# text = "Cute Cats"
# font = cv2.putText(img, text, (470,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=outlier_color, thickness=1, lineType=cv2.LINE_AA)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#endregion

#region Part 1 -- Marker
# img = np.zeros((300, 300, 3), np.uint8)
#
# cv2.drawMarker(img, (50, 150),color=(255,0,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=5)
# cv2.drawMarker(img, (100, 150),color=(0,255,0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=5)
# cv2.drawMarker(img, (150, 150),color=(255,0,0), markerType=cv2.MARKER_DIAMOND, markerSize=20, thickness=5)
# cv2.drawMarker(img, (200, 150),color=(255,0,0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=20, thickness=5)
# cv2.drawMarker(img, (250, 150),color=(255,0,0), markerType=cv2.MARKER_SQUARE, markerSize=20, thickness=5)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#endregion

#region Part 3 Marker Exercise
# img = np.zeros((300, 300, 3), np.uint8)
# corners = [(50,50),(200,200),(150,250)]
# for pt in corners:
#     cv2.drawMarker(img, pt,  (0, 255, 0), 2)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#endregion

#region Part 4 -- Arrowed Line
# img = np.zeros((400, 400, 3), np.uint8)
#
# pt_1 = (100,100)
# pt_2 = (300,300)
# color = (255,0,0)
#
# cv2.arrowedLine(img, pt_1, pt_2, color, 2, tipLength=0.1)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#endregion

#region Part 5 -- Face Detection

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture_cam = cv2.VideoCapture(0)

while True:
    success, img = capture_cam.read()
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(img_grey, 1.1, 19)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('openCv', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_cam.release()
cv2.destroyAllWindows()


face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
capture_cam = cv2.VideoCapture(0)

while True:
    success, img = capture_cam.read()
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(img_grey, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'eye', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('openCv', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_cam.release()
cv2.destroyAllWindows()

#endregion

#region Part 6

#endregion



# Promoted to Learning Lab:
#   core.vision_classic.faces  +  Image Studio → Effects ▸ Classic Vision
# Live webcam loops stay in this session file (desktop OpenCV), not Streamlit.