import os
import sys
import cv2
import pickle
import numpy as np
import logging as log
import time
import csv
from numpy import dot
from numpy.linalg import norm
from argparse import ArgumentParser
from inference import Network
from detector import Detector
from reidentificator import Reidentifier
from collections import namedtuple
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt
from pyqtgraph import ImageView
from gui import MainWindow, ImageWidget
from person import Person
import pandas as pd

# Global variables
TARGET_DEVICE = 'CPU'
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
is_async_mode = True

# Flag to control background thread
KEEP_RUNNING = True

DELAY = 5

selected_areas={}
people=[]

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-e", "--detector", required=True, type=str,
                        help="Path to an xml file with a trained detection model.")
    parser.add_argument("-r", "--reidentificator", required=True, type=str,
                        help="Path to an xml file with a trained reidentification model.")
    parser.add_argument("-i", "--input_v", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-k", "--k_figures", required=True, type=str,
                        help="Path to folder with photos of known people")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    global TARGET_DEVICE, is_async_mode
    args = parser.parse_args()
    if args.device:
        TARGET_DEVICE = args.device
    return parser


def check_args():
    # ArgumentParser checks the device
    global TARGET_DEVICE
    if 'MULTI' not in TARGET_DEVICE and TARGET_DEVICE not in accepted_devices:
        print("Unsupported device: " + TARGET_DEVICE)
        sys.exit(2)
    elif 'MULTI' in TARGET_DEVICE:
        target_devices = TARGET_DEVICE.split(':')[1].split(',')
        for multi_device in target_devices:
            if multi_device not in accepted_devices:
                print("Unsupported device: " + TARGET_DEVICE)
                sys.exit(2)

def performance_counts(perf_count):
    """
    print information about layers of the model.

    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))


def ssd_out(frame, result, known_figures, reidentifier, names, selected_areas, del_time):
    """
    Parse SSD output.

    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    global people
    person = []
    ischanged = False
    best_name=''
    best_val=0
    drawn = False
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            person.append([xmin, ymin, xmax, ymax])
            #Recognize person            
            if len(known_figures)>0:
                crop_img = frame[ymin:ymax, xmin:xmax]
                #Get the object blob
                detected = reidentifier.reidentify_single(crop_img)
                if detected is not None:
                    for name in names:
                        params=known_figures[name]
                        for param in params:
                            #Comparing known pictures bolb to object blob, using cosine similarity
                            cos_sim=dot(param[0], detected[0])/(norm(param[0])*norm(detected[0]))
                            if cos_sim > best_val:
                                best_val = cos_sim
                                if best_name != name:
                                    best_name = name
                    if best_val>0.7 and not drawn:
                        cv2.rectangle(frame, (xmin, ymin+10), (xmax, ymax), (148, 0, 211), 3)
                        cv2.putText(frame, best_name, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (148, 0, 211))
                        drawn = True   
            if not drawn:
                cv2.putText(frame, 'Unknown', (xmin, ymin-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 55, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 3) 

            if len(selected_areas) > 0:
                #check the area where person is
                for p in person:
                    # area_of_person gives area of the detected person
                    area_of_person = (p[2] - p[0]) * (p[3] - p[1])
                    for key in selected_areas:
                        selected_region = selected_areas[key]
                        x_max = max(p[0], selected_region[0])
                        x_min = min(p[2], selected_region[0] + selected_region[2])
                        y_min = min(p[3], selected_region[1] + selected_region[3])
                        y_max = max(p[1], selected_region[1])
                        point_x = x_min - x_max
                        point_y = y_min - y_max
                        # area_of_intersection gives area of intersection of the
                        # detected person and the selected area
                        area_of_intersection = point_x * point_y
                        if point_x < 0 or point_y < 0:
                            continue
                        else:
                            if area_of_person <= area_of_intersection:
                                if drawn:
                                    for person in people:
                                        if person.name == best_name:
                                            if person.__dict__[key+'_time'] != 0: 
                                                person.__dict__[key+'_time'] += del_time
                                            else:
                                                person.__dict__[key+'_time'] += 0.01
                                            ischanged = True
    return frame, ischanged
    

def main(gui):
    """
    Load the network and parse the SSD output.

    :return: None
    """
    global DELAY
    global SIG_CAUGHT
    global KEEP_RUNNING
    global TARGET_DEVICE
    global is_async_mode
    global people

    args = build_argparser().parse_args()
    check_args()

    # Flag for the input image
    single_image_mode = False
    cur_request_id = 0

    # Initialise the class
    detect_network = Network()
    reid_network = Network()

    # Checks for live feed
    if args.input_v == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input_v.endswith('.jpg') or args.input_v.endswith('.bmp')  or args.input_v.endswith('.png'):
        single_image_mode = True
        input_stream = args.input_v

    # Checks for video file
    else:
        input_stream = args.input_v
        assert os.path.isfile(args.input_v), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input_v)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    
    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)  

    reidentifier_f = Reidentifier(reidentifier=args.reidentificator, folder=args.k_figures, device=args.device,
                                        ext=args.cpu_extension, infer_network=reid_network) 
    known_figures={}
    names=[]

    if os.path.exists(r'pickles\known_figures.pkl'):
        with open(r'pickles\known_figures.pkl', 'rb') as f:
           known_figures = pickle.load(f)
        with open(r'pickles\names.pkl', 'rb') as f:
           names = pickle.load(f)
                                                                                        #SUPER NOTES add the upgrade for pickles from gui
    else:
        known_figures, names = reidentifier_f.reidentify_fldr()
        with open(r'pickles\known_figures.pkl', 'wb') as f:
            pickle.dump(known_figures, f)
        with open(r'pickles\names.pkl', 'wb') as f:
            pickle.dump(names, f)

    cur_request_id = 0
 
    # read csv file
    df = pd.read_csv(r"pickles\time.csv")
    keys = list(df.keys())   
    for i in range(1, len(keys)):
        argum = {}
        for j in range(len(df)):
            argum[df['area_name'][j]+'_time'] = df[keys[i]][j]
        per = Person(name=keys[i],**argum)
        people.append(per)


    detector = Detector(detector=args.detector, device=args.device, ext=args.cpu_extension, 
                                infer_network=detect_network, cur_request_id=cur_request_id)

    selected_region = None

    gui.start(people)
    ttl_frames = 0
    prev_time = 0
    del_time = 0

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        ttl_frames += 1
        result = detector.detect(frame)
        key_pressed = cv2.waitKey(1)
        if result is not None:
            cur_request_id = 0
            if key_pressed == 99 or gui.change == True:
                if len(selected_areas) != 0:
                    for key in selected_areas.keys():
                        cords = selected_areas[key]
                        #color = tuple(np.random.choice(range(256), size=3))
                        #print(color)
                        cv2.rectangle(frame, (cords[0], cords[1]),
                            (cords[0] + cords[2], cords[1] + cords[3]), (0, 0, 255), 2)
                        cv2.putText(frame, key, (int((cords[0]+cords[2])/2), int((cords[1]+cords[3])/2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                # Give operator chance to change the area
                # Select rectangle from left upper corner, dont display crosshair
                # SUPER NOTEs make user able to exit
                ROI = cv2.selectROI("Choose the area", frame, True, False)
                print("Assembly Area Selection: -x = {}, -y = {}, -w = {}, -h = {}".format(ROI[0], ROI[1], ROI[2], ROI[3]))
                roi_x = ROI[0]
                roi_y = ROI[1]
                roi_w = ROI[2]
                roi_h = ROI[3]
                cv2.destroyAllWindows()

                cv2.rectangle(frame, (roi_x, roi_y),
                      (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
                selected_region = [roi_x, roi_y, roi_w, roi_h]
                gui.change = False
                selected_areas[gui.area_name] = selected_region

            ttl_time = ttl_frames/fps
            del_time = ttl_time - prev_time
            
            frame, changed = ssd_out(frame, result, known_figures, reidentifier_f, names, selected_areas, del_time)
            
            prev_time = ttl_time
            #update pickles data
            if len(gui.new_person) == 2:
                names.append(gui.new_person[0])
                items = []
                for image in gui.new_person[1]:
                    shape_img = cv2.imread(image)
                    shape_img = reidentifier_f.reidentify_single(shape_img)
                    items.append(shape_img)
                known_figures[gui.new_person[0]] = tuple(items)
                with open(r'pickles\known_figures.pkl', 'wb') as f:
                    pickle.dump(known_figures, f)
                with open(r'pickles\names.pkl', 'wb') as f:
                    pickle.dump(names, f)
                print(known_figures)
                gui.new_person = []

            if changed:
                gui.update_people(people)
                #update and push changes to csv file   
                df1 = pd.read_csv(r"pickles\time.csv")   
                for person in people:
                    for count in range(0,len(df1)):
                        df1[person.name][count] = list(person.__dict__.values())[count+1]
                    df1.to_csv(r'C:\Users\Mat\Desktop\people-counter-python-master\time.csv')
            gui.grab_images(frame)
            if cv2.waitKey(25) & 0xFF == ord('q') or gui.close_inf == True:
                break
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
    cap.release()
    detect_network.clean()
    reid_network.clean()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = MainWindow()
    gui.show()
    main(gui)
    exit(0)
