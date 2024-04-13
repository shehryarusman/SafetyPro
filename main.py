from app import TextDetectionApp
from mainWindow import MainWindow
import tkinter as tk

# Used for drawing over screen
import sys
from PyQt5.QtWidgets import QApplication
from screeninfo import get_monitors
from mainWindow import MainWindow
from threading import Thread

def runTextDetection(hide_func):
    app = TextDetectionApp(hide_func)
    # root.mainloop()


if __name__ == "__main__":
    qApp = QApplication(sys.argv)
    qWindow = MainWindow(get_monitors()[0].width, get_monitors()[0].height)
    # root = tk.Tk()
    textDecThread = Thread(target=runTextDetection, args=(qWindow.addRect,))
    textDecThread.start()
    qWindow.show()
    qApp.exec_()


