# USB camera display using PyQt and OpenCV, from iosoft.blog
# Copyright (c) Jeremy P Bentham 2019
# Please credit iosoft.blog if you use the information or software in it

VERSION = "Cam_display v0.10"

import sys, time, threading, cv2
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer, QPoint, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel, QPushButton
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout, QDesktopWidget, QInputDialog
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor, QLineEdit
import queue as Queue

IMG_SIZE    = 1280,720         # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 2                # Scaling factor for display image
DISP_MSEC   = 50                # Delay between display cycles
CAP_API     = cv2.CAP_ANY       # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE    = 0                 # Zero for automatic exposure
TEXT_FONT   = QFont("Courier", 10)

camera_num  = 1                 # Default camera (first in list)    # Queue to hold images
capturing   = True              # Flag to indicate capturing

# Grab images from the camera (separate thread)

# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

# Main window
class MainWindow(QMainWindow):
    text_update = pyqtSignal(str)
    # Create main window
    def __init__(self, parent=QMainWindow):
        super().__init__()
        self.change = False
        self.close_inf = False
        self.image_queue = Queue.Queue() 
        self.central = QWidget(self)
        self.textbox = QTextEdit(self.central)
        self.textbox.setFont(TEXT_FONT)
        self.textbox.setMinimumSize(600, 100)
        self.text_update.connect(self.append_text)
        sys.stdout = self
        print("Camera number %u" % camera_num)
        print("Image size %u x %u" % IMG_SIZE)
        if DISP_SCALE > 1:
            print("Display scale %u:1" % DISP_SCALE)
        self.frame_c = None
        self.vlayout = QVBoxLayout()        # Window layout
        self.displays = QHBoxLayout()
        self.disp = ImageWidget(self)    
        self.displays.addWidget(self.disp)
        self.vlayout.addLayout(self.displays)
        self.label = QLabel(self)
        self.vlayout.addWidget(self.label)
        self.vlayout.addWidget(self.textbox)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.area_name = ''
        self.trackw = TrackWindow()
        self.mainMenu = self.menuBar()      # Menu bar
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        setAction = QAction('&Set Area', self)
        setAction.setShortcut('Ctrl+S')
        setAction.triggered.connect(self.change_window)
        currentPresence = QAction('&See current presence', self)
        currentPresence.setShortcut('Ctrl+P')
        currentPresence.triggered.connect(self.track_presence)
        self.fileMenu = self.mainMenu.addMenu('&Menu')
        self.fileMenu.addAction(exitAction)
        self.fileMenu.addAction(setAction)
        self.fileMenu.addAction(currentPresence)

    # Start image capture & display
    def start(self):
        self.timer = QTimer(self)           # Timer to trigger display
        self.timer.timeout.connect(lambda: 
                    self.show_image(self.image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC) 
    def grab_images(self, frame):
        if frame is not None and self.image_queue.qsize() < 2:
            self.image_queue.put(frame)
            self.frame_c = frame
        else:
            time.sleep(DISP_MSEC / 1000.0)

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        disp_size = int((QDesktopWidget().screenGeometry(-1).width())/2),int((QDesktopWidget().screenGeometry(-1).height())/2)
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size, 
                             interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1], 
                      disp_bpl, IMG_FORMAT)
        display.setImage(qimg)

    # Handle sys.stdout.write: update text display
    def write(self, text):
        self.text_update.emit(str(text))
    def flush(self):
        pass

    # Append to text display
    def append_text(self, text):
        cur = self.textbox.textCursor()     # Move cursor to end of text
        cur.movePosition(QTextCursor.End) 
        s = str(text)
        while s:
            head,sep,s = s.partition("\n")  # Split line at LF
            cur.insertText(head)            # Insert text at cursor
            if sep:                         # New line if LF
                cur.insertBlock()
        self.textbox.setTextCursor(cur)     # Update visible cursor

    def change_window(self, event):
        text, okPressed = QInputDialog.getText(self, "Enter the name for the Area","Name of the area:")
        if okPressed and text != '':
            self.change = True
            self.area_name = text


    def closeEvent(self, event):
        self.close_inf = True 
        event.accept()
    
    def track_presence(self):
        self.trackw.show()
        
class TrackWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.central = QWidget(self)
        self.vlayout = QVBoxLayout()        # Window layout
        self.label = QLabel(self)
        self.add_button = QPushButton('Add new workers')
        self.vlayout.addWidget(self.label)
        self.add_button.setMinimumSize(600, 50)
        self.vlayout.addWidget(self.add_button)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.add_button.clicked.connect(self.on_click)
    def on_click(self):
        pass