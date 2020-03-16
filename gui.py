# USB camera display using PyQt and OpenCV, from iosoft.blog
# Copyright (c) Jeremy P Bentham 2019
# Please credit iosoft.blog if you use the information or software in it

VERSION = "Cam_display v0.10"

import sys, time, threading, cv2, os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer, QPoint, pyqtSignal, pyqtSlot, QModelIndex
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel, QPushButton
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout, QDesktopWidget, QInputDialog, QLineEdit, QFileSystemModel, QTreeView, QScrollArea
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor, QLineEdit, QPixmap
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
        self.shoot = False
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
        self.shoot_button = QPushButton('Shoot')
        self.areas_button = QPushButton('See areas stats')
        self.disp = ImageWidget(self)    
        self.displays.addWidget(self.disp)
        self.buttons.addWidget(self.set_button)
        self.buttons.addWidget(self.cp_button)
        self.buttons.addWidget(self.shoot_button)
        self.buttons.addWidget(self.areas_button)
        self.vlayout.addLayout(self.displays)
        self.vlayout.addLayout(self.buttons)
        self.label = QLabel(self)
        self.vlayout.addWidget(self.label)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.area_name = ''
        self.mainMenu = self.menuBar()  
        self.ars = Areas()
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
        setAction.triggered.connect(self.change_window)
        areasStats = QAction('&See areas stats', self)
        areasStats.setShortcut('Ctrl+A')
        areasStats.triggered.connect(self.show_areas)
        self.fileMenu = self.mainMenu.addMenu('&Menu')
        self.fileMenu.addAction(exitAction)
        self.fileMenu.addAction(setAction)
        self.fileMenu.addAction(currentPresence)
        self.fileMenu.addAction(areasStats)
        self.set_button.clicked.connect(self.change_window)
        self.cp_button.clicked.connect(self.track_presence)
        self.shoot_button.clicked.connect(self.change_shot)
        self.areas_button.clicked.connect(self.show_areas)

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

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        disp_size = 1280,720
        disp_bpl = disp_size[0] * 3
        '''
        if scale > 1:
            img = cv2.resize(img, disp_size, 
                             interpolation=cv2.INTER_CUBIC)
        '''
        qimg = QImage(img.data, disp_size[0], disp_size[1], 
                      disp_bpl, IMG_FORMAT)
        display.setImage(qimg)

    # Add area
    def change_window(self, event):
        text, okPressed = QInputDialog.getText(self, "Area","Name of the area:", QLineEdit.Normal, "")
        if okPressed and text != '':
            self.change = True
            self.area_name = text
            if not os.path.exists(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+text):
                os.mkdir(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+text)    

    def closeEvent(self, event):
        self.close_inf = True 
        event.accept()
    
    def change_shot(self):
        self.shoot = not self.shoot
        if self.shoot_button.text() == "Shoot":
            self.shoot_button.setText("Stop shooting")
        else:
            self.shoot_button.setText("Shoot")

    def track_presence(self):
        self.trackw.show()

    def update_people(self, ppl, stats=False, row=False, column=False):
        self.trackw.people = ppl
        if column:
            self.trackw.upadte_column(self.trackw.people[-1])   
        if row:
            self.trackw.update_row()
        if stats:
            self.trackw.update_stats()
    
    def show_areas(self):
        self.ars.show()

    def update_areas(self, areas):
        ar_keys = list(areas.keys())
        for ind in range(len(ar_keys)):
            self.ars.update_button(ind, ar_keys[ind])
        
class TrackWindow(QMainWindow):
    def __init__(self, people):
        super().__init__()
        self.people = people
        self.setWindowTitle("Tracking stats")  
        self.result = []
        self.central = QWidget(self)
        self.vlayout = QVBoxLayout() 
        self.button_names = []       
        for person in self.people:
            self.upadte_column(person)
        self.add_button = QPushButton('Add new people')
        self.add_button.setMinimumSize(int((QDesktopWidget().screenGeometry(-1).width())/4), int((QDesktopWidget().screenGeometry(-1).height())/15))
        self.vlayout.addWidget(self.add_button)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.add_button.clicked.connect(self.add_person)

    def add_person(self):
        self.name_person, okPressed = QInputDialog.getText(self, "Name","Name of a person:",QLineEdit.Normal, "")
        if okPressed and self.name_person != '':
            self.window = QWidget()
            self.window.setWindowTitle('Doubleclick your image')
            okButtons = QHBoxLayout()
            windowLayout = QVBoxLayout()
            '''
            self.ok_button = QPushButton('Ok')
            self.cancel_button = QPushButton('Cancel')
            okButtons.addWidget(self.ok_button)
            okButtons.addWidget(self.cancel_button)
            self.ok_button.setEnabled(False)
            self.ok_button.setCheckable(True)
            '''
            self.model = QFileSystemModel()
            self.model.setRootPath('')
            self.tree = QTreeView()
            self.tree.setModel(self.model)
            self.tree.setIndentation(20)
            self.tree.setSortingEnabled(True)
            self.tree.setWindowTitle("Dir View")
            self.tree.setMinimumSize(int((QDesktopWidget().screenGeometry(-1).width())/2), int((QDesktopWidget().screenGeometry(-1).height())/2))
            windowLayout.addWidget(self.tree)
            windowLayout.addLayout(okButtons)
            self.window.setLayout(windowLayout)
            self.tree.doubleClicked.connect(self.on_treeView_clicked)
            self.window.show()
        else:
            print('Enter something')

    @pyqtSlot(QModelIndex)    
    def on_treeView_clicked(self, index):
        indexItem = self.model.index(index.row(), 0, index.parent())
        values = []
        fileName = self.model.fileName(indexItem)
        filePath = self.model.filePath(indexItem)
        values.append(filePath)
        self.result.append(self.name_person)               
        self.result.append(values)
        
    def update_stats(self):
        for person in self.people:
            ind = str(self.people.index(person))
            for key, value in person.__dict__.items():
                parametr = getattr(self, key+ind)
                if key  != 'name':     
                    time_val = float(value.split(' ')[-2])
                    indx = value.find(str(time_val))
                    parametr.setText(key+' time: '+value[:indx]+str(int(time_val))+value[value.find('sec')-1:]) 
    
    def upadte_column(self, person):
            ind = self.people.index(person)
            setattr(self, 'button'+str(ind), QVBoxLayout())
            self.button_names.append('button'+str(ind))
            if ind%2 == 0:
                setattr(self, 'buttons'+str(ind), QHBoxLayout())
            for key, value in person.__dict__.items():
                setattr(self, key+str(ind), QLabel(self))
                parametr = getattr(self, key+str(ind))
                if key  != 'name':
                    time_val = float(value.split(' ')[-2])
                    indx = value.find(str(time_val))
                    parametr.setText(key+' time: '+value[:indx]+str(int(time_val))+value[value.find('sec')-1:])
                else:
                    parametr.setText(value)
                but_ind = getattr(self, 'button'+str(ind))
                but_ind.addWidget(parametr)
                if ind%2 == 0:
                    box_lay = getattr(self, 'buttons'+str(ind))
                else:
                    box_lay = getattr(self, 'buttons'+str(ind-1))
                box_lay.addLayout(but_ind)
                self.vlayout.insertLayout(-2,box_lay)

    def update_row(self):
        key = list(self.people[-1].__dict__.keys())[-1]
        count = 0 
        for name in self.button_names:
            but_ind = getattr(self, name) 
            setattr(self, key+str(count), QLabel(self))
            parametr = getattr(self, key+str(count))
            parametr.setText(key+' time: 0 hrs 0 min 0.0 sec')
            but_ind.addWidget(parametr)  
            count += 1


class Areas(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Areas")  
        self.central = QWidget(self)
        self.vlayout = QVBoxLayout()      
        self.add_button = QPushButton('Upload saved zones')
        self.add_button.setMinimumSize(int((QDesktopWidget().screenGeometry(-1).width())/4), int((QDesktopWidget().screenGeometry(-1).height())/15))
        self.vlayout.addWidget(self.add_button)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.add_button.clicked.connect(self.upadte_status)
        self.status = False

    def upadte_status(self):
        self.status = True

    def update_button(self, ind, text):
        try:
            parametr = getattr(self, 'button'+str(ind))
        except AttributeError:
            setattr(self, 'button'+str(ind), QPushButton(text))
            parametr = getattr(self, 'button'+str(ind))
            parametr.clicked.connect(lambda:self.show_areawindow(ind, text))
            if ind%2 == 0:
                setattr(self, 'buttons'+str(ind), QHBoxLayout())
                self.but_ind = getattr(self, 'buttons'+str(ind))
                self.but_ind.addWidget(parametr)
                self.vlayout.addLayout(self.but_ind)
            else:
                self.but_ind = getattr(self, 'buttons'+str(ind-1))
                self.but_ind.addWidget(parametr)
    
    def show_areawindow(self, ind, name):
        try:
            win = getattr(self, 'window'+str(ind)) 
        except AttributeError:
            setattr(self, 'window'+str(ind), AreaWindow(name)) 
            win = getattr(self, 'window'+str(ind)) 
        win.add_picture()
        win.show()

class AreaWindow(QMainWindow):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.setWindowTitle(self.name + " Area")  
        self.central = QWidget()
        self.vlayout = QVBoxLayout() 
        self.scroll_lay = QVBoxLayout() 
        self.scroll_lay = QScrollArea()
        self.central.setLayout(self.vlayout)
        #Scroll Area Properties
        self.scroll_lay.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_lay.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_lay.setWidgetResizable(True)
        self.scroll_lay.setWidget(self.central)

        self.setCentralWidget(self.scroll_lay)
        self.resize(500, 700)
    
    def add_picture(self):
        os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+self.name)
        for filename in os.listdir(os.getcwd()):
            ind = os.listdir(os.getcwd()).index(filename)
            setattr(self, 'label'+str(ind),  QLabel(self)) 
            win = getattr(self, 'label'+str(ind)) 
            pixmap = QPixmap(filename)
            pixmap = pixmap.scaled(150, 300, transformMode=Qt.SmoothTransformation)
            win.setPixmap(pixmap)
            if ind%3 == 0:
                setattr(self, 'pics'+str(ind), QHBoxLayout())
                img_ind = getattr(self, 'pics'+str(ind))
                img_ind.addWidget(win)
                self.vlayout.addLayout(img_ind)
            else:
                if (ind-1)%3 == 0:
                    img_ind = getattr(self, 'pics'+str(ind-1))
                if (ind-2)%3 == 0:
                    img_ind = getattr(self, 'pics'+str(ind-2))
                img_ind.addWidget(win)
        os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master')
