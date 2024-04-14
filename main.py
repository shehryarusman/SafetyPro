from app import TextDetectionApp
from mainWindow import MainWindow
import tkinter as tk

# Used for drawing over screen
import sys
from PyQt5.QtWidgets import QApplication
from screeninfo import get_monitors
from mainWindow import MainWindow
from threading import Thread

def runTextDetection(hide_func, clear_func, update_func):
    app = TextDetectionApp(hide_func, clear_func, update_func)

if __name__ == "__main__":
    qApp = QApplication(sys.argv)
    qWindow = MainWindow(get_monitors()[0].width, get_monitors()[0].height)
    textDecThread = Thread(target=runTextDetection, args=(qWindow.addRect,qWindow.clearRects, qWindow.update))
    textDecThread.start()
    qWindow.show()
    qApp.exec_()
    textDecThread.join()
    exit()


