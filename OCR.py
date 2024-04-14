import os
from pathlib import Path
import sys
from datetime import datetime
import time
import threading
from threading import Thread

import tkinter as tk
from tkinter import Canvas

from screeninfo import get_monitors


from PyQt5 import QtWidgets, QtCore, QtGui

from PyQt5.QtCore import pyqtSignal, QObject, QRect
from PyQt5.QtGui import QPainter, QColor, QImage, QPixmap

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

from PIL import Image, ImageTk

from mss import mss

import cv2
import numpy
import pytesseract

from text_classification import TextDetectionApp

textDetectApp = TextDetectionApp()

def tesseract_location(root):
    try:
        pytesseract.pytesseract.tesseract_cmd = root
    except FileNotFoundError:
        print("Please double check the Tesseract file directory or ensure it's installed.")
        sys.exit(1)


class RateCounter:
    
    def __init__(self):
        self.start_time = None
        self.iterations = 0

    def start(self):
        """
        Starts a time.perf_counter() and sets it in the self.start_time attribute

        :return: self
        """
        self.start_time = time.perf_counter()
        return self

    def increment(self):
        """
        Increases the self.iterations attribute
        """
        self.iterations += 1

    def rate(self):
        """
        Returns the iterations/seconds
        """
        elapsed_time = (time.perf_counter() - self.start_time)
        return self.iterations / elapsed_time


class VideoStream:
    def __init__(self, src=1):  # Default to the primary monitor
        self.sct = mss()
        self.mon = self.sct.monitors[src]  # Use src to select the monitor
        self.frame = self.capture_screen()
        self.stopped = False

    def capture_screen(self):
        # Create a new MSS instance every time to avoid thread-local storage issues
        with mss() as sct:
            sct_img = sct.grab(self.mon)
            frame = numpy.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
        return frame
        # # Use 'mon' dictionary directly with mss to capture the screen
        # sct_img = self.sct.grab(self.mon)
        # # Convert the image to a format suitable for processing and display
        # frame = numpy.array(sct_img)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
        # return frame

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            self.frame = self.capture_screen()

    def get_video_dimensions(self):
        """
        Return the dimensions of the video based on the monitor's resolution.
        """
        return self.mon['width'], self.mon['height']

    def stop_process(self):
        self.stopped = True


class OCR:
    """
    Class for creating a pytesseract OCR process in a dedicated thread

    `Attributes:`
        boxes: Data output from pytesseract (includes bounding boxes, confidence, and string for detected test)
        stopped: bool indicating whether the process has been stopped
        exchange: The a reference to VideoStream class where frames are grabbed and processed
        language: language code(s) for detecting custom languages in pytesseract
        width: Horizontal dimension of the VideoStream frame
        height: Vertical dimension of the VideoSteam frame
        crop_width: Horizontal crop amount if OCR is to be performed on a smaller area
        crop_height: Vertical crop amount if OCR is to be performed on a smaller area

    `Methods:`
        start()
            Creates a thread targeted at the ocr process
        set_exchange(VideoStream)
            Sets the self.exchange attribute with a reference to VideoStream class
        set_language(language)
            Sets the self.language parameter
        ocr()
            Creates a process where frames are continuously grabbed from the exchange and processed by pytesseract OCR
        set_dimensions(width, height, crop_width, crop_height):
            Sets the dimensions attributes
        stop_process()
            Sets the self.stopped attribute to True
    """

    # def __init__(self, exchange: VideoStream, language=None):
    def __init__(self):
        self.boxes = None
        self.stopped = False
        self.exchange = None
        self.language = None
        self.width = None
        self.height = None
        self.crop_width = None
        self.crop_height = None
        self.frame = None

    def start(self):
        """
        Creates a thread targeted at the ocr process
        :return: self
        """
        Thread(target=self.ocr, args=()).start()
        return self

    def set_exchange(self, video_stream):
        """
        Sets the self.exchange attribute with a reference to VideoStream class
        :param video_stream: VideoStream class
        """
        self.exchange = video_stream

    def set_language(self, language):
        """
        Sets the self.language parameter
        :param language: language code(s) for detecting custom languages in pytesseract
        """
        self.language = language

    def ocr(self):
        """
        Creates a process where frames are continuously grabbed from the exchange and processed by pytesseract OCR.
        Output data from pytesseract is stored in the self.boxes attribute.
        """
        while not self.stopped:
            if self.exchange is not None:  # Defends against an undefined VideoStream reference
                frame = self.exchange.frame
                self.frame = frame

                # # # CUSTOM FRAME PRE-PROCESSING GOES HERE # # #
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                # # # # # # # # # # # # # # # # # # # #

                frame = frame[self.crop_height:(self.height - self.crop_height),
                              self.crop_width:(self.width - self.crop_width)]

                self.boxes = pytesseract.image_to_data(frame, lang=self.language, output_type=pytesseract.Output.DICT)

    def set_dimensions(self, width, height, crop_width, crop_height):
        """
        Sets the dimensions attributes

        :param width: Horizontal dimension of the VideoStream frame
        :param height: Vertical dimension of the VideoSteam frame
        :param crop_width: Horizontal crop amount if OCR is to be performed on a smaller area
        :param crop_height: Vertical crop amount if OCR is to be performed on a smaller area
        """
        self.width = width
        self.height = height
        self.crop_width = crop_width
        self.crop_height = crop_height

    def stop_process(self):
        """
        Sets the self.stopped attribute to True and kills the ocr() process
        """
        self.stopped = True


def capture_image(frame, captures=0):
    """
    Capture a .jpg during CV2 video stream. Saves to a folder /images in working directory.

    :param frame: CV2 frame to save
    :param captures: (optional) Number of existing captures to append to filename

    :return: Updated number of captures. If capture param not used, returns 1 by default
    """
    cwd_path = os.getcwd()
    Path(cwd_path + '/images').mkdir(parents=False, exist_ok=True)

    now = datetime.now()
    # Example: "OCR 2021-04-8 at 12:26:21-1.jpg"  ...Handles multiple captures taken in the same second
    name = "OCR " + now.strftime("%Y-%m-%d") + " at " + now.strftime("%H:%M:%S") + '-' + str(captures + 1) + '.jpg'
    path = 'images/' + name
    cv2.imwrite(path, frame)
    captures += 1
    print(name)
    return captures


def views(mode: int, confidence: int):
    """
    View modes changes the style of text-boxing in OCR.

    View mode 1: Draws boxes on text with >75 confidence level

    View mode 2: Draws red boxes on low-confidence text and green on high-confidence text

    View mode 3: Color changes according to each word's confidence; brighter indicates higher confidence

    View mode 4: Draws a box around detected text regardless of confidence

    :param mode: view mode
    :param confidence: The confidence of OCR text detection

    :returns: confidence threshold and (B, G, R) color tuple for specified view mode
    """
    conf_thresh = None
    color = None

    if mode == 1:
        conf_thresh = 75  # Only shows boxes with confidence greater than 75
        color = (0, 0, 0, 255)  # BLACK
        color = (0, 255, 0, 255)  # Green

    if mode == 2:
        conf_thresh = 0  # Will show every box
        if confidence >= 50:
            color = (0, 255, 0, 255)  # Green
        else:
            color = (0, 0, 255, 255)  # Red

    if mode == 3:
        conf_thresh = 0  # Will show every box
        color = (int(float(confidence)) * 2.55, int(float(confidence)) * 2.55, 0, 255)

    if mode == 4:
        conf_thresh = 0  # Will show every box
        color = (0, 0, 255, 255)  # Red

    return conf_thresh, color


def put_ocr_boxes(boxes, frame, height, width, crop_width=0, crop_height=0, view_mode=1):
    """
    Draws text bounding boxes at tesseract-specified text location. Also displays compatible (ascii) detected text
    Note: ONLY works with the output from tesseract image_to_data(); image_to_boxes() uses a different output format

    :param boxes: output tuple from tesseract image_to_data() containing text location and text string
    :param numpy.ndarray frame: CV2 display frame destination
    :param height: Frame height
    :param crop_width: (Default 0) Horizontal frame crop amount if OCR was performed on a cropped frame
    :param crop_height: (Default 0) Vertical frame crop amount if OCR was performed on a cropped frame
    :param view_mode: View mode to specify style of bounding box

    :return: CV2 frame with bounding boxes, and output text string for detected text
    """

    if view_mode not in [1, 2, 3, 4]:
        raise Exception("A nonexistent view mode was selected. Only modes 1-4 are available")

    rects = []
    text = ''  # Initializing a string which will later be appended with the detected text
    transparent_img = numpy.zeros((height, width, 4), dtype=numpy.uint8)
    if boxes is not None:  # Defends against empty data from tesseract image_to_data
        badBoxIndices = textDetectApp.detect_text(boxes)
        for i in badBoxIndices:  # Next three lines turn data into a list
            x, y, w, h = int(boxes['left'][i]), int(boxes['top'][i]), int(boxes['width'][i]), int(boxes['height'][i])
            rects.append(QRect(x, y, w, h))
            conf = boxes['conf'][i]
            x += crop_width  # If tesseract was performed on a cropped image we need to 'convert' to full frame
            y += crop_height

            conf_thresh, color = views(view_mode, int(float(conf)))
            # if int(float(conf)) > conf_thresh:
            cv2.rectangle(transparent_img, (x, y), (w + x, h + y), color, -1)
        # for i, box in enumerate(boxes.splitlines()):  # Next three lines turn data into a list
        #     box = box.split()
        #     if i != 0:
        #         if len(box) == 12:
        #             x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
        #             conf = box[10]
        #             word = box[11]
        #             x += crop_width  # If tesseract was performed on a cropped image we need to 'convert' to full frame
        #             y += crop_height

        #             conf_thresh, color = views(view_mode, int(float(conf)))
        #             if int(float(conf)) > conf_thresh:
        #                 cv2.rectangle(transparent_img, (x, y), (w + x, h + y), color, thickness=1)
        #                 text = text + ' ' + word
            
    return transparent_img, text, rects


def put_crop_box(frame: numpy.ndarray, width: int, height: int, crop_width: int, crop_height: int):
    """
    Simply draws a rectangle over the frame with specified height and width to show a crop zone

    :param numpy.ndarray frame: CV2 display frame for crop-box destination
    :param int width: Width of the CV2 frame
    :param int height: Height of the CV2 frame
    :param int crop_width: Horizontal crop amount
    :param int crop_height: Vertical crop amount

    :return: CV2 display frame with crop box added
    """
    cv2.rectangle(frame, (crop_width, crop_height), (width - crop_width, height - crop_height),
                  (255, 0, 0, 255), thickness=1)
    return frame


def put_rate(frame: numpy.ndarray, rate: float) -> numpy.ndarray:
    """
    Places text showing the iterations per second in the CV2 display loop.

    This is for demonstrating the effects of multi-threading.

    :param frame: CV2 display frame for text destination
    :param rate: Iterations per second rate to place on image

    :return: CV2 display frame with rate added
    """

    cv2.putText(frame, "{} Iterations/Second".format(int(rate)),
                (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    return frame


def put_language(frame: numpy.ndarray, language_string: str) -> numpy.ndarray:
    """
    Places text showing the active language(s) in current OCR display

    :param numpy.ndarray frame: CV2 display frame for language name destination
    :param str language_string: String containing the display language name(s)

    :returns: CV2 display frame with language name added
    """
    cv2.putText(frame, language_string,
                (10, 65), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    return frame

class OverlayWindow(QMainWindow):
    update_image_signal = pyqtSignal(QImage)  # Define a signal that takes a QImage

    def __init__(self, video_stream, ocr_processor):
        super().__init__()
        self.video_stream = video_stream
        self.ocr_processor = ocr_processor
        self.initUI()
        self.image = None  # This will hold the QImage to be displayed

        self.update_image_signal.connect(self.update_image)  # Connect signal to slot

    def initUI(self):
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setGeometry(self.video_stream.mon['left'], self.video_stream.mon['top'],
                         self.video_stream.mon['width'], self.video_stream.mon['height'])
        self.show()

    def paintEvent(self, event):
        if self.image:
            qp = QPainter(self)
            qp.drawImage(self.rect(), self.image)
        

    def update_image(self, img):
        self.image = img
        self.update()

    def updateMask(self, rectangles):
        self.mask_pixmap = QtGui.QPixmap(self.size())
        self.mask_pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(self.mask_pixmap)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 0, 100, 150), QtCore.Qt.SolidPattern))
        for rect in rectangles:
            painter.drawRect(rect)
        painter.end()
        self.update()




class MainWindow(QMainWindow):

    def __init__(self, w, h):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(
            QtWidgets.QStyle.alignedRect(
                QtCore.Qt.LeftToRight, QtCore.Qt.AlignCenter,
                QtCore.QSize(w, h),
                QtWidgets.qApp.desktop().availableGeometry()
            )
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self.mask_pixmap = None
        self.mask_rect = QtCore.QRect()
        # Need to add rectangles with locations to cover
        # then update painter and it should be over the word we want

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.mask_pixmap:
            qp.drawPixmap(self.rect(), self.mask_pixmap)
        qp.end()

    def updateMask(self, rectangles):
        self.mask_pixmap = QPixmap(self.size())
        self.mask_pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(self.mask_pixmap)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 0, 100, 150), QtCore.Qt.SolidPattern))
        for rect in rectangles:
            painter.drawRect(rect)
        painter.end()
        self.update()
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            print("Exiting")
            QtWidgets.qApp.quit()
            exit()

    def addRect(self, x, y, w, h):
        self.rectangles.append(QtCore.QRect(x, y, w, h))

    def clearRects(self):
        return
        self.rectangles.clear()

def update_frame(video_stream, ocr, overlay_window, view_mode):
    while not video_stream.stopped and not ocr.stopped:
        #frame = video_stream.frame
        frame = ocr.frame
        frame = cv2.resize(frame, (1920//2, 1080//2), interpolation=cv2.INTER_AREA)
        img_hi = video_stream.mon['height']
        img_wi = video_stream.mon['width']
        # print(ocr.boxes)
        frame, text, rects = put_ocr_boxes(ocr.boxes, frame, img_hi, img_wi, view_mode=view_mode)

        # Convert frame to QImage
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        h, w, ch = frame_rgba.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgba.data, 1920, 1080, bytes_per_line, QImage.Format_RGBA8888)


        #overlay_window.updateMask(rects)
        overlay_window.update_image(qt_image)
        #overlay_window.update_image(qt_image)
        #overlay_window.update_image_signal.emit(qt_image)  # Emit the signal

        QtCore.QThread.msleep(150)  # Refresh rate


def ocr_stream(crop: list[int, int], source: int = 0, view_mode: int = 1, language=None):
    video_stream = VideoStream(source).start()
    img_wi, img_hi = video_stream.get_video_dimensions()

    if crop is None:
        cropx, cropy = (0, 0)  # Default crop
    else:
        cropx, cropy = crop[0], crop[1]
        if cropx > img_wi or cropy > img_hi or cropx < 0 or cropy < 0:
            cropx, cropy = 0, 0

    ocr = OCR().start()
    ocr.set_exchange(video_stream)
    ocr.set_language(language)
    ocr.set_dimensions(img_wi, img_hi, cropx, cropy)

    app = QApplication(sys.argv)
    overlay_window = OverlayWindow(video_stream,ocr)
    text_dec_thread = Thread(target=update_frame, args=(video_stream, ocr, overlay_window, view_mode))
    text_dec_thread.start()
    overlay_window.show()
    app.exec_()
    text_dec_thread.join()

    video_stream.stop_process()
    ocr.stop_process()