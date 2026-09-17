import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# region Part 1 - create empty image
#
# img = np.zeros((1200, 1200, 3), np.uint8)
# # print(img)
#
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()

# endregion

# region Part 2 - draw line

# img = np.zeros((1200, 1200, 3), np.uint8)
# r, c, _ = img.shape
# print(img.shape)

# pt_1 = (0,0)
# pt_2 =r, c
#
# color = (255, 0, 0)
#
# thickness = 4

# pt_1 = (100,0)  # سطر | ستون
# pt_2 = (c-100,500)
# color = 255
# thickness = 4

# pt_1 = (1100,600)  # سطر | ستون
# pt_2 = (100,600)
# color = 255
# thickness = 4
#
# cv2.line(img, pt_1, pt_2, color, thickness)
#
# plt.imshow(img)
# plt.axis('off')
# plt.show()
#
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("./output/1.png", img)


# endregion

# region Part 3 - rectangle

# img = cv2.imread("images/cute-cats.bmp")
#
# rows, cols, c = img.shape
#
# pt_1 = (100, 280)
# pt_2 = (370, rows -4)
#
# color = (128,128,128)
#
# cv2.rectangle(img, pt_1, pt_2, color,  thickness=1)
# cv2.imshow("cut cat", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# endregion

# region Part 4 - rectangle

# img = cv2.imread('images/cute-cats.bmp')
# rows, cols, c = img.shape
#
# pt_1 = (100, 280)
# pt_2 = (370, rows-1)
# color = (128/255, 128/255, 128/255)
#
# img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
# mask = np.zeros_like(img)
#
# cv2.rectangle(mask, pt_1, pt_2, color, -1)
#
# img_result = img + 0.2* mask
# cv2.imshow('result', img_result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# endregion

# region Part 6 - circle
# img = np.zeros((1000, 1000, 3), np.uint8)
#
# center = (500, 500)
# radius = 250
# color = (0,0,255)
# thickness = 3
#
# cv2.circle(img, center, radius, color, thickness)
#
# center2 = (0, 0)
# radius2 = 250
# color2 = (0,0,255)
# thickness2 = 3
#
# cv2.circle(img, center2, radius2, color2, thickness2)
#
#
# center3 = (0, 0)
# radius3 = 250
# color3 = (0,0,255)
# thickness3 = 3
#
# cv2.circle(img, center3, radius3, color3, thickness=-thickness3)
#
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# endregion

# region Part 7 - Polylines
# img = np.zeros((600, 600, 3), np.uint8)
#
# pt_set1 = np.array([[100, 100], [200, 100], [200, 200], [100, 100]], dtype=np.int32)
# pt_set2 = np.array([[300, 100], [400, 200], [300, 200]], dtype=np.int32)
#
# color = (0, 0, 255)
#
# cv2.polylines(img, [pt_set1, pt_set2], True, color, 2)
#
# cv2.imshow("polylines", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# endregion

# region Part 8 - fill ploy

# img = np.zeros((600, 600, 3), np.uint8)
#
# pt_set1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.int32)
# pt_set2 = np.array([[300, 100], [400, 200], [300, 200]], dtype=np.int32)
#
# color = (0, 0, 255)
#
# cv2.fillPoly(img, [pt_set1, pt_set2], color)
#
# cv2.imshow("polylines", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# endregion


# region exercise

# img = np.zeros((400, 400, 3), np.uint8)
#
# pts = np.array([
#     [200, 5],
#     [230, 170],
#     [350, 170],
#     [250, 250],
#     [280, 370],
#     [200, 300],
#     [120, 370],
#     [150, 250],
#     [170, 170],
#     [170, 170]
# ], dtype= np.uint32)
#
# cv2.fillPoly(img, [pts], (0, 250, 0))
#
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# endregion

# region Part 8 - video
# cap = cv2.VideoCapture(0) # 0 دوربین اصلی سیستم
#
# cap_2 = cv2.VideoCapture("video/bouncing-ball-25fps.mp4")
#
# cap_5 = cv2.VideoCapture("https://opencv.org/wp-content/uploads/2025/02/Example-Video.mp4")
#
# print(cap_2.isOpened())
#
# ret, frame = cap_2.read()
# print(ret, frame.shape)
#
#
# cv2.imshow("frame", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# endregion

# region Part 9 - webcam
# Guarded: uncomment to try local webcam (desktop OpenCV). Do not leave this
# running as the default when opening this session file.
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cap.release()

# endregion
