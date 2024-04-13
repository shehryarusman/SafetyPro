import sys

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from screeninfo import get_monitors

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

        self.rectangles = []
        # Need to add rectangles with locations to cover
        # then update painter and it should be over the word we want

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        # Change color here to actually cover after debugging is done
        qp.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 150), QtCore.Qt.SolidPattern))
        qp.drawRects(self.rectangles)
        qp.end()
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            print("Exiting")
            QtWidgets.qApp.quit()

    def addRect(self, x, y, w, h):
        self.rectangles.append(QtCore.QRect(x, y, w, h))

    def clearRects(self):
        self.rectangles.clear

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(get_monitors()[0].width, get_monitors()[0].height)
    window.show()
    app.exec_()