# USB camera display using PyQt and OpenCV, from iosoft.blog
# Copyright (c) Jeremy P Bentham 2019
# Please credit iosoft.blog if you use the information or software in it

VERSION = "Cam_display v0.10"

import sys, time, threading, cv2
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer, QPoint, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel, QPushButton
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout, QDesktopWidget, QInputDialog, QLineEdit
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
        self.trackw = None
        self.frame_c = None
        self.vlayout = QVBoxLayout()      
        self.displays = QHBoxLayout()
        self.buttons = QHBoxLayout()
        self.set_button = QPushButton('Set Area')
        self.cp_button = QPushButton('See current presence')
        self.disp = ImageWidget(self)    
        self.displays.addWidget(self.disp)
        self.buttons.addWidget(self.set_button)
        self.buttons.addWidget(self.cp_button)
        self.vlayout.addLayout(self.displays)
        self.vlayout.addLayout(self.buttons)
        self.label = QLabel(self)
        self.vlayout.addWidget(self.label)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.area_name = ''
        self.mainMenu = self.menuBar()  
        self.new_person = [] 
        self.setWindowTitle("Surveillance in progress")  
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
        self.set_button.clicked.connect(self.change_window)
        self.cp_button.clicked.connect(self.track_presence)

    # Start image capture & display
    def start(self, people):
        self.timer = QTimer(self)           
        self.timer.timeout.connect(lambda: 
                    self.show_image(self.image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC) 
        self.trackw = TrackWindow(people)

    def grab_images(self, frame):
        if frame is not None and self.image_queue.qsize() < 2:
            self.image_queue.put(frame)
            self.frame_c = frame
        else:
            time.sleep(DISP_MSEC / 1000.0)
        if len(self.trackw.result) == 2:
            self.new_person = self.trackw.result[:]
            self.trackw.result = []

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

    # Add area
    def change_window(self, event):
        text, okPressed = QInputDialog.getText(self, "Area","Name of the area:", QLineEdit.Normal, "")
        if okPressed and text != '':
            self.change = True
            self.area_name = text

    def closeEvent(self, event):
        self.close_inf = True 
        event.accept()
    
    def track_presence(self):
        self.trackw.show()

    def update_people(self, ppl):
        self.trackw.people = ppl
        self.trackw.update_stats()

class TrackWindow(QMainWindow):
    def __init__(self, people):
        super().__init__()
        self.people = people
        self.setWindowTitle("Tracking stats")  
        self.result = []
        self.central = QWidget(self)
        self.vlayout = QVBoxLayout()        
        for person in self.people:
            ind = self.people.index(person)
            setattr(self, 'button'+str(ind), QVBoxLayout())
            if ind%2 == 0:
                 setattr(self, 'buttons'+str(ind), QHBoxLayout())
            for key, value in person.__dict__.items():
                setattr(self, key+str(ind), QLabel(self))
                parametr = getattr(self, key+str(ind))
                if key  != 'name':
                    parametr.setText(key+' time: '+str(int(value))) 
                else:
                    parametr.setText(str(value))
                but_ind = getattr(self, 'button'+str(ind))
                but_ind.addWidget(parametr)
                if ind%2 == 0:
                    box_lay = getattr(self, 'buttons'+str(ind))
                else:
                    box_lay = getattr(self, 'buttons'+str(ind-1))
                box_lay.addLayout(but_ind)
                self.vlayout.addLayout(box_lay)

        self.add_button = QPushButton('Add new workers')
        self.add_button.setMinimumSize(int((QDesktopWidget().screenGeometry(-1).width())/4), int((QDesktopWidget().screenGeometry(-1).height())/15))
        self.vlayout.addWidget(self.add_button)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.add_button.clicked.connect(self.add_person)

    def add_person(self):
        name_person, okPressed = QInputDialog.getText(self, "Name","Name of a person:",QLineEdit.Normal, "")
        if okPressed and name_person != '':
            self.result.append(name_person)
            values = []
            pic = QInputDialog(self)
            print(pic.cancelButtonText)
            picture, ok = pic.getText(self, "Enter the picture directory","Picture directory:",QLineEdit.Normal, "")
            while ok and picture != '':
                values.append(picture)
                picture, ok = pic.getText(self, "Enter the picture directory","Picture directory:",QLineEdit.Normal, "")
            self.result.append(values)
    
    def update_stats(self):
        for person in self.people:
            ind = str(self.people.index(person))
            for key, value in person.__dict__.items():
                parametr = getattr(self, key+ind)
                if key  != 'name':
                    parametr.setText(key+' time: '+str(int(value))) 

            