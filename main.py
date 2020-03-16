import os
import sys
import cv2
import pickle
import numpy as np
import logging as log
import time
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
from area import Area
from multiprocessing import Process
from threading import Thread, Lock
import pandas as pd
import telebot
import queue as Queue
from flask import Flask, request

# Global variables
TARGET_DEVICE = 'CPU'
accepted_devices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
is_async_mode = True
colors = [(128, 0, 0), (128, 128, 0), (255, 0, 0)]

# Flag to control background thread
KEEP_RUNNING = True

DELAY = 5

selected_areas = {}
people = []

reidentifier_f = None
detector = None

known_figures={}
names=[]
msg_id = None
show = False
ischanged = False
queue = Queue.Queue()
last_images = []
key_fldrs = []
lock = Lock()
bot = None
gui = None
fldr = ''
limg = ''
listen_name = False
double_auth = False
tname=''

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


def draw(frame, result, shoot):
    """
    Parse SSD output.

    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    global people
    global areas
    global reidentifier_f
    global known_figures
    global names
    global queue
    global last_img
    global lock
    global selected_areas
    global ischanged
    per_man = []
    ischanged = False
    best_name=''
    best_val=0
    start_saving = False
    crop_imgs = []
    crop_img = None
    drawn_list = []
    best_names = []
    for obj in result[0][0]:
        drawn = False
        # Draw bounding box for object when it's probability is more than the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            per_man.append([xmin, ymin, xmax, ymax])
            crop_img = frame[ymin:ymax, xmin:xmax]
            crop_imgs.append(crop_img)
            if shoot:
                os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people') 
                cv2.imwrite('seen'+str(np.random.choice(range(50)))+'.jpg', crop_img) 
                os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master')
            #Recognize person            
            if len(known_figures)>0:
                #Get the object blob
                detected = reidentifier_f.reidentify_single(crop_img)
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
                        best_names.append(best_name)
            if not drawn:
                cv2.putText(frame, 'Unknown', (xmin, ymin-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 55, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 3) 
            drawn_list.append(drawn) 
            if len(selected_areas) > 0:
                start_saving = True                        
    return frame, start_saving, per_man, drawn_list, crop_imgs, best_names

def save_img(per_man, drawn, crop_img, best_names, del_time):
    global bot
    global reidentifier_f
    global last_images
    global key_fldrs
    global msg_id
    global people
    global selected_areas
    global ischanged
    print(per_man)
    print(drawn)
    for i in range(len(per_man)):
        # area_of_person gives area of the detected person
        area_of_person = (per_man[i][2] - per_man[i][0]) * (per_man[i][3] - per_man[i][1])
        for key in selected_areas:
            selected_region = selected_areas[key]
            rx_max = max(per_man[i][0], selected_region[0])
            rx_min = min(per_man[i][2], selected_region[0] + selected_region[2])
            ry_min = min(per_man[i][3], selected_region[1] + selected_region[3])
            ry_max = max(per_man[i][1], selected_region[1])

            point_x = rx_min - rx_max
            point_y = ry_min - ry_max
            # area_of_intersection gives area of intersection of the detected person and the selected area
            area_of_intersection = point_x * point_y
            if point_x < 0 or point_y < 0:
                continue
            else:
                if area_of_person <= area_of_intersection:
                    if drawn[i]:
                        for person in people:
                            if person.name == best_names[i]:
                                val = person.__dict__[key]      
                                time_val = float(val.split(' ')[4])
                                indx = val.find(str(time_val))
                                time_val += del_time
                                mins = int(val.split(' ')[2])
                                if time_val > 60.0:
                                    mins += 1
                                    person.__dict__[key] = val[:6]+str(mins)+val[7:indx]+'0.0'+val[val.find('sec')-1:]
                                else:
                                    person.__dict__[key] = val[:indx]+str(time_val)+val[val.find('sec')-1:]
                                    if mins > 60:
                                        hours = str(int(val.split(' ')[0])+1)
                                        person.__dict__[key] = hours+ val[1:6]+'0'+val[9:]     
                                    ischanged = True
                    else:
                        crop_blb = reidentifier_f.reidentify_single(crop_img[i])
                        exists = False
                        for picture in os.listdir(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+key):
                            cv_pic = cv2.imread(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+key+chr(92)+picture)
                            cv_pic = reidentifier_f.reidentify_single(cv_pic)
                            cos_sim=dot(cv_pic[0], crop_blb[0])/(norm(cv_pic[0])*norm(crop_blb[0])) 
                            if cos_sim > 0.4:
                                exists = True
                                break
                        if exists == False:
                            rand_name = 'seen'+str(np.random.choice(range(100)))+'.jpg'
                            cv2.imwrite(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+key+chr(92)+rand_name, crop_img[i])
                            last_images.append(rand_name)
                            key_fldrs.append(key)

def bot_action():
    global show
    global bot
    global last_images
    global key_fldrs
    bot = telebot.TeleBot('1027363216:AAHE3VPx7NlivMVzQvaL-8zyFuf8_4U52a8')
    '''
    server = Flask(__name__)
    print(11)
    # Empty webserver index, return nothing, just http 200
    @server.route('/', methods=['GET', 'HEAD'])
    def index():
        return ''
    print(12)
    # Process webhook calls
    @server.route('/1027363216:AAHE3VPx7NlivMVzQvaL-8zyFuf8_4U52a8/', methods=['POST'])
    def webhook():
        if flask.request.headers.get('content-type') == 'application/json':
            json_string = flask.request.get_data().decode('utf-8')
            update = telebot.types.Update.de_json(json_string)
            bot.process_new_updates([update])
            return ''
        else:
            flask.abort(403)
    # Start flask server
    print(111)
    server.run(host='0.0.0.0' ,
            port=8443,
            ssl_context=(r'C:\Program Files\OpenSSL-Win64\bin\webhook_cert.pem', r'C:\Program Files\OpenSSL-Win64\bin\webhook_pkey.pem'))
   '''
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        markup = telebot.types.ReplyKeyboardMarkup(row_width=2)
        show_lst = telebot.types.KeyboardButton('Show last')
        markup.add(show_lst)
        bot.send_message(message.chat.id, "What should I do?", reply_markup=markup)

    @bot.message_handler(func=lambda message: True,content_types=['text'])
    def show_images(message):
        global fldr
        global limg
        global listen_name 
        global double_auth
        global known_figures
        global names
        global tname
        global reidentifier_f
        global people
        global gui

        msg_id = message.chat.id
        if message.text.lower() == 'show last':
            if len(last_images)>0:
                fldr = key_fldrs.pop()
                limg = last_images.pop()
                photo = open(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+fldr+chr(92)+limg, 'rb')
                bot.send_message(msg_id, 'Unknown person detected:')
                bot.send_photo(msg_id, photo)
                markup = telebot.types.ReplyKeyboardRemove(selective=False)
                bot.send_message(msg_id, '~~~', reply_markup=markup)
                markup = telebot.types.ReplyKeyboardMarkup(row_width=2)
                knowbtn = telebot.types.KeyboardButton('I know who it is!')
                noideabtn = telebot.types.KeyboardButton('I have no idea')
                noisebtn = telebot.types.KeyboardButton('Picture is a noise')
                markup.add(knowbtn, noideabtn, noisebtn)
                bot.send_message(msg_id, "Choose what to do with this image:", reply_markup=markup)
            else:
                bot.send_message(msg_id, "No people detected yet")
                bot.send_message(msg_id, "What should I do?")

        elif message.text.lower() == 'i know who it is!':
            if fldr != '' and listen_name == False:
                markup = telebot.types.ReplyKeyboardRemove(selective=False)
                bot.send_message(msg_id, '~~~', reply_markup=markup)
                bot.send_message(msg_id, "Tell me the name of this person")
                listen_name = True
            else:
                bot.send_message(msg_id, "Wrong input please refer to the buttons")
                bot.send_message(msg_id, "What should I do?")

        elif message.text.lower() == 'i have no idea' or message.text.lower() == 'picture is a noise':
            if fldr != '' and listen_name == False:
                markup = telebot.types.ReplyKeyboardRemove(selective=False)
                bot.send_message(msg_id, 'Okay, image is removed', reply_markup=markup)
                os.remove(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+fldr+chr(92)+limg)
                fldr = ''
                limg = ''
                mark = telebot.types.ReplyKeyboardMarkup(row_width=2)
                show_lst = telebot.types.KeyboardButton('Show last')
                mark.add(show_lst)
                bot.send_message(msg_id, "What should I do?", reply_markup=mark)
            else:
                bot.send_message(msg_id, "Wrong input please refer to the buttons")
                bot.send_message(msg_id, "What should I do?")
            
        elif message.text != '' and message.text.lower() != 'i know who it is!' and listen_name :
            bot.send_message(msg_id, "Name specified by you:"+message.text)
            tname = message.text
            double_auth = True
            markup = telebot.types.ReplyKeyboardRemove(selective=False)
            bot.send_message(msg_id, '~~~', reply_markup=markup)
            mku = telebot.types.ReplyKeyboardMarkup(row_width=2)
            yesbtn = telebot.types.KeyboardButton('Yes')
            nobtn = telebot.types.KeyboardButton('No')
            mku.add(yesbtn, nobtn)
            bot.send_message(msg_id, "Are you willing to continue with this name?", reply_markup=mku)
            listen_name = False

        elif message.text.lower() == 'yes':
            if double_auth:
                if tname not in names and tname!='':
                    known_figures[tname] = []
                    names.append(tname)
                # add person to csv
                df = pd.read_csv(r"pickles\time.csv")
                vals = []
                vals_dict = {}
                for j in range(1,len(people[0].__dict__.keys())):
                    vals.append('0 hrs 0 min 0.0 sec')
                    vals_dict[list(people[0].__dict__.keys())[j]] = '0 hrs 0 min 0.0 sec'
                df[tname] = vals
                new_p = Person(name = tname, **vals_dict)                                       #separate function
                people.append(new_p)
                df.to_csv(r"pickles\time.csv", index = False)
                gui.update_people(people, column = True)
                print(limg)
                shape_img = cv2.imread(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people'+chr(92)+fldr+chr(92)+limg)
                print(shape_img)
                shape_img = reidentifier_f.reidentify_single(shape_img)
                print(shape_img)
                if shape_img is not None:
                    known_figures[tname].append(shape_img)
                print(known_figures)
                print(names)
                with open(r'pickles\known_figures.pkl', 'wb') as f:
                    pickle.dump(known_figures, f)
                with open(r'pickles\names.pkl', 'wb') as f:
                    pickle.dump(names, f)
                tname = ''
                fldr = ''
                limg = '' 
                double_auth = False
                markup = telebot.types.ReplyKeyboardRemove(selective=False)
                bot.send_message(msg_id, '~~~', reply_markup=markup)
                mark = telebot.types.ReplyKeyboardMarkup(row_width=2)
                show_lst = telebot.types.KeyboardButton('Show last')
                mark.add(show_lst)
                bot.send_message(msg_id, "What should I do?", reply_markup=mark)

        elif message.text.lower() == 'no':
            if double_auth:
                tname = ''
                markup = telebot.types.ReplyKeyboardRemove(selective=False)
                bot.send_message(msg_id, '~~~', reply_markup=markup)
                markup = telebot.types.ReplyKeyboardMarkup(row_width=2)
                knowbtn = telebot.types.KeyboardButton('I know who it is!')
                noideabtn = telebot.types.KeyboardButton('I have no idea')
                noisebtn = telebot.types.KeyboardButton('Picture is a noise')
                markup.add(knowbtn, noideabtn, noisebtn)
                bot.send_message(msg_id, "Choose what to do with this image:", reply_markup=markup)
                double_auth = False
            else:
                bot.send_message(msg_id, "Wrong input please refer to the buttons")
                bot.send_message(msg_id, "What should I do?")
        else:
            bot.send_message(msg_id, "Wrong input please refer to the buttons")
            bot.send_message(msg_id, "What should I do?")

    bot.polling(none_stop=False, interval=1)
    
    '''
    bot.remove_webhook()
    print(3)
    time.sleep(2)
    bot.set_webhook(url='https://matveitss.pythonanywhere.com/:8443/1027363216:AAHE3VPx7NlivMVzQvaL-8zyFuf8_4U52a8/',
                certificate=open(r'C:\Program Files\OpenSSL-Win64\bin\webhook_cert.pem', 'r'))
    print(4)
    '''
def main(gui_p):
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
    global selected_areas
    global reidentifier_f
    global detector
    global known_figures
    global names
    global show
    global gui
    global ischanged
    gui = gui_p
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
        input_stream = 'rtsp://admin:admin@192.168.43.223:554/1'   

    # Checks for input image
    elif args.input_v.endswith('.jpg') or args.input_v.endswith('.bmp')  or args.input_v.endswith('.png'):
        single_image_mode = True
        input_stream = args.input_v

    # Checks for video file
    else:
        input_stream = args.input_v
        assert os.path.isfile(args.input_v), "Specified input file doesn't exist"
    cap = cv2.VideoCapture(input_stream)
    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    initial_w = 1280
    initial_h = 720  
    fps = cap.get(cv2.CAP_PROP_FPS)  
    reidentifier_f = Reidentifier(reidentifier=args.reidentificator, folder=args.k_figures, device=args.device,
                                        ext=args.cpu_extension, infer_network=reid_network) 
    if os.path.exists(r'pickles\known_figures.pkl'):
        with open(r'pickles\known_figures.pkl', 'rb') as f:
           known_figures = pickle.load(f)
        with open(r'pickles\names.pkl', 'rb') as f:
           names = pickle.load(f)
    else:
        known_figures, names = reidentifier_f.reidentify_fldr()
        with open(r'pickles\known_figures.pkl', 'wb') as f:
            pickle.dump(known_figures, f)
        with open(r'pickles\names.pkl', 'wb') as f:
            pickle.dump(names, f)

    cur_request_id = 0
 
    # read csv file with zones
    df = pd.read_csv(r"pickles\zones.csv")
    os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master\seen people')
    keys = list(df.keys()) 
    for j in range(len(df)):
        selected_areas[df[keys[0]][j]] = [df[keys[1]][j], df[keys[2]][j], df[keys[3]][j], df[keys[4]][j]]
        if df[keys[0]][j] not in os.listdir(os.getcwd()):
            os.mkdir(df[keys[0]][j])         
    os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master')

    # read csv with times
    df = pd.read_csv(r"pickles\time.csv")
    keys = list(df.keys())   
    for i in range(1, len(keys)):
        argum = {}
        for j in range(len(df)):
            argum[df['area_name'][j]] = df[keys[i]][j]
        per = Person(name=keys[i],**argum)
        people.append(per)

    detector = Detector(detector=args.detector, device=args.device, ext=args.cpu_extension, 
                                infer_network=detect_network, cur_request_id=cur_request_id)

    bot_process = Thread(target=bot_action, args = ())
    bot_process.start()

    selected_region = None

    gui.start(people)
    ttl_frames = 0
    prev_time = 0
    del_time = 0
    min_counter = 0
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        frame = cv2.resize(frame, (initial_w,initial_h))
        ttl_frames += 1
        result = detector.detect(frame)
        key_pressed = cv2.waitKey(1)
        if result is not None:
            cur_request_id = 0
            if key_pressed == 99 or gui.change == True:
                if len(selected_areas) != 0:
                    for key in selected_areas.keys():
                        cords = selected_areas[key]
                        color = colors[np.random.choice(range(3))]
                        cv2.rectangle(frame, (cords[0], cords[1]),
                            (cords[0] + cords[2], cords[1] + cords[3]), color, 2)
                        cv2.putText(frame, key, (int((cords[0]+cords[2])/2), int((cords[1]+cords[3])/2)), cv2.FONT_HERSHEY_COMPLEX, 1, color)
                # Give operator chance to change the area
                # Select rectangle from left upper corner, dont display crosshair
                # SUPER NOTEs make user able to exit
                ROI = cv2.selectROI("Choose the area", frame, True, False)
                roi_x = ROI[0]
                roi_y = ROI[1]
                roi_w = ROI[2]
                roi_h = ROI[3]
                cv2.destroyAllWindows()
                selected_region = [roi_x, roi_y, roi_w, roi_h]
                gui.change = False
                selected_areas[gui.area_name] = selected_region
                gui.update_areas(selected_areas)
                #add new zone to pickles file
                df = pd.read_csv(r"pickles\time.csv")
                if not gui.area_name in list(df['area_name']):
                    vals=[gui.area_name]
                    for i in range(len(people)):
                        vals.append('0 hrs 0 min 0.0 sec')
                    df.loc[len(df.index)]=vals
                    df.to_csv(r"pickles\time.csv", index=False)
                    for person in people:
                        person.__setattr__(gui.area_name, '0 hrs 0 min 0.0 sec')
                    gui.update_people(people, row = True)
                df1 = pd.read_csv(r"pickles\zones.csv")
                sr = selected_region.copy()
                sr.insert(0, gui.area_name)
                if not gui.area_name in list(df1['area_name']):
                    df1.loc[len(df1.index)] = sr
                else:
                    for row in range(len(df1.index)):
                        if df1['area_name'][row] == gui.area_name:
                            df1.loc[row] = sr
                df1.to_csv(r"pickles\zones.csv", index=False)
                
            ttl_time = ttl_frames/fps
            del_time = ttl_time - prev_time
            min_counter += del_time
            frame, save, per_man, drawn, crop_img, best_names = draw(frame, result, gui.shoot)
            if save:
                saving_process = Thread(target=save_img, args=(per_man, drawn, crop_img, best_names, del_time))
                saving_process.start()
                saving_process.join()
            os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master')
            prev_time = ttl_time
            #update pickles data for new person
            if len(gui.trackw.result) == 2:
                for i in range(len(gui.trackw.result[1])):
                    if i == 0 and gui.trackw.result[0] not in names:
                        known_figures[gui.trackw.result[0]] = []
                        names.append(gui.trackw.result[0])
                        # add person to csv
                        df = pd.read_csv(r"pickles\time.csv")
                        vals = []
                        vals_dict = {}
                        for j in range(1,len(people[0].__dict__.keys())):
                            vals.append('0 hrs 0 min 0.0 sec')
                            vals_dict[list(people[0].__dict__.keys())[j]] = '0 hrs 0 min 0.0 sec'
                        df[gui.trackw.result[0]] = vals
                        new_p = Person(name = gui.trackw.result[0], **vals_dict)
                        people.append(new_p)
                        df.to_csv(r"pickles\time.csv", index = False)
                        gui.update_people(people, column = True)
                    shape_img = cv2.imread(gui.trackw.result[1][i])
                    shape_img = reidentifier_f.reidentify_single(shape_img)
                    if shape_img is not None:
                        exists = False
                        for j in range(len(known_figures[gui.trackw.result[0]])): 
                            if np.array_equal(known_figures[gui.trackw.result[0]][j], shape_img):
                                exists = True
                        if exists == False:
                            known_figures[gui.trackw.result[0]].append(shape_img)
                    else:
                        print('Wrong image: ' + gui.trackw.result[1][i])
                with open(r'pickles\known_figures.pkl', 'wb') as f:
                    pickle.dump(known_figures, f)
                with open(r'pickles\names.pkl', 'wb') as f:
                    pickle.dump(names, f)
                gui.trackw.result = []

            if gui.ars.status:
                gui.update_areas(selected_areas)
                gui.ars.status = False

            if ischanged:
                gui.update_people(people, stats = True)
                #update and push changes to csv file  
                if min_counter >= 2: 
                    df1 = pd.read_csv(r"pickles\time.csv")   
                    for person in people:
                        for count in range(0,len(df1)):
                            df1[person.name][count] = list(person.__dict__.values())[count+1]
                    df1.to_csv(r"pickles\time.csv", index=False)
                    min_counter = 0

            gui.grab_images(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or gui.close_inf == True:
                break
        if single_image_mode:
            os.chdir(r'C:\Users\Mat\Desktop\people-counter-python-master\img')
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