# made by Tir

from tkinter import *
import os
import cv2
import sys
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont, ImageTk
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from tkinter import messagebox
sys.path.append('sort')
from sort.sort import *
if os.name == 'posix':
    import tflite_runtime.interpreter as tflite
elif os.name == 'nt':
    import tensorflow as tf
    assert tf.__version__.startswith('2')
    tf.get_logger().setLevel('ERROR')


if os.name == 'nt':
    faces_tflite = 'efficientdet-lite-faces.tflite'
    faces_labels = 'labels.txt'
    persons_tflite = 'efficientdet-lite-persons.tflite'
    persons_labels = 'labels-persons.txt'
elif os.name == 'posix':
    # nu merge cu edgetpu
    faces_tflite = 'efficientdet-lite-faces_edgetpu.tflite'
    faces_labels = 'labels.txt'
    persons_tflite = 'efficientdet-lite-persons_edgetpu.tflite'
    persons_labels = 'labels-persons.txt'

mot_tracker = Sort()

tflite_model_name = persons_tflite
tflite_labels_name = persons_labels
main_x = 320.5
main_y = 240.5

if os.name == 'posix':
    import board
    import pwmio
    from adafruit_motor import servo
    pwm_pan_x = pwmio.PWMOut(board.PWM1, duty_cycle=0, frequency=50)
    pwm_tilt_y = pwmio.PWMOut(board.PWM2, duty_cycle=0, frequency=50)
    pwm_pan = servo.Servo(pwm_pan_x)
    pwm_tilt = servo.Servo(pwm_tilt_y)
    pwm_pan.angle = 90
    pwm_tilt.angle = 90


    def move_servo(servo, delta, max, angles, sens):
        error = abs(delta)
        sign = sens * error / delta
        if error <= 40:
            pass
        elif 40 < error <= (max + 20) / 3 and angles[0] < servo.angle + 3 < angles[1]:
            servo.angle += 2 * sign
        elif (max + 20) / 3 < error <= max and angles[0] < servo.angle + 7 < angles[1]:
            servo.angle += 5 * sign
        else:
            pass


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


def draw_box(img, objs, scale_factor, labels, track_bbs_ids, fps):
    """
    Deseneaza chenarele in jurul obiectelor si afiseaza probabilitatea si id-ul.

    Parameters
    -------
    img: numpy.ndarray
        imaginea pe care trebuie desenat
    objs: list
        obiectele prezente in cadru
    scale_factor: float
        raportul dintre dimensiunile intitiale si cele noi
    labels: dict
        etichetele utilizate
    track_bbs_ids: list
        lista de id-uri si pozitia corespunzatoare
    """

    fps_text = "FPS: {:.2f}".format(fps)
    color = (0, 255, 0)
    cv2.putText(img, fps_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    for obj in objs:
        bbox = obj.bbox
        start_point = (int(bbox.xmin * scale_factor), int(bbox.ymin * scale_factor))
        end_point = (int(bbox.xmax * scale_factor), int(bbox.ymax * scale_factor))
        cx = (start_point[0]+end_point[0])//2
        cy = (start_point[1]+end_point[1])//2
        if len(objs) > 0:
            # selectare pozi
            pozi = None
            for i in range(len(objs)):
                if ((bbox.xmin + bbox.xmax) / 2 == (objs[i].bbox.xmin + objs[i].bbox.xmax) / 2) and \
                        ((bbox.ymin + bbox.ymax) / 2 == (objs[i].bbox.ymin + objs[i].bbox.ymax) / 2):
                    pozi = i
            try:
                cv2.putText(
                    img,
                    '%s %d %.2f' % (labels.get(obj.id, obj.id), track_bbs_ids[pozi][4], obj.score),
                    (int(bbox.xmin * scale_factor - 5), int(bbox.ymin * scale_factor - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.rectangle(img, start_point, end_point, color, 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
            except IndexError:
                pass


def get_target(track_bbs_ids, id=0):
    global main_x, main_y
    center_x = 320.5
    center_y = 240.5
    delta_x = 0
    delta_y = 0
    try:
        main_x = (track_bbs_ids[id][0] + track_bbs_ids[id][2]) / 2
        main_y = (track_bbs_ids[id][1] + track_bbs_ids[id][3]) / 2
        delta_x = round(center_x - main_x, 2)
        delta_y = round(center_y - main_y, 2)
        return delta_x, delta_y
    except IndexError:
        pass


class App:
    def __init__(self, parent):
        self.parent = parent
        self.label = Label(self.parent, text="In ce context va fi utilizat algoritmul?", font=('DejavuSans', 12))
        self.label.pack(pady=20)
        self.button1 = Button(self.parent, text='Interior', font=('DejavuSans', 12), command=self.new_win_1)
        self.button1.pack(pady=20)
        self.button2 = Button(self.parent, text='Exterior', font=('DejavuSans', 12), command=self.new_win_2)
        self.button2.pack(pady=20)

    def new_win_1(self):
        fpsmediu = []
        global tflite_labels_name, tflite_model_name
        cap = cv2.VideoCapture(0)
        if os.name == 'posix':
            pwm_pan.angle = 90
            pwm_tilt.angle = 90

        def on_closing_a():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                cap.release()
                print(clicked.get(), " interior, fps mediu: ", np.mean(fpsmediu))
                new_win.destroy()

        def change_model():
            global tflite_labels_name, tflite_model_name, labels, interpreter
            if clicked.get() == 'Person detection':
                tflite_labels_name = persons_labels
                tflite_model_name = persons_tflite
            elif clicked.get() == 'Face detection':
                tflite_labels_name = faces_labels
                tflite_model_name = faces_tflite
            labels = read_label_file(tflite_labels_name)
            if os.name == 'posix':
                interpreter = tflite.Interpreter(tflite_model_name,
                                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            elif os.name == 'nt':
                interpreter = tf.lite.Interpreter(tflite_model_name)
            interpreter.allocate_tensors()

        def callback(selection):
            change_model()

        new_win = Toplevel(self.parent)
        new_win.title("Interior")
        new_win.geometry("750x640")
        new_win.configure(bg="black")
        new_win.protocol("WM_DELETE_WINDOW", on_closing_a)
        m1 = Label(new_win, bg="black")
        m1.pack(pady=20)
        options = [
            "Person detection",
            "Face detection"
        ]
        clicked = StringVar(new_win)
        clicked.set(options[0])
        drop = OptionMenu(new_win, clicked, *options, command=callback)
        drop.pack(pady=20)
        fps_start_time = 0
        fps = 0
        while True:
            labels = read_label_file(tflite_labels_name)
            if os.name == 'posix':
                interpreter = tflite.Interpreter(tflite_model_name,
                                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            elif os.name == 'nt':
                interpreter = tf.lite.Interpreter(tflite_model_name)
            interpreter.allocate_tensors()

            display_width = 640
            try:
                img = cap.read()[1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                _, scale = common.set_resized_input(
                    interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
                interpreter.invoke()
                objs = detect.get_objects(interpreter, score_threshold=0.55, image_scale=scale)
                fps_end_time = time.time()
                time_diff = fps_end_time - fps_start_time
                fps = 1 / time_diff
                fps_start_time = fps_end_time
                boxes = []
                centers = []
                for obj in objs:
                    boxes.append(np.array([obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax, obj.score]))
                    centers.append([(obj.bbox.xmin + obj.bbox.xmax)/2, (obj.bbox.ymin + obj.bbox.ymax)/2])
                if objs:
                    track_bbs_ids = mot_tracker.update(np.array(boxes))
                    a_x, a_y = get_target(track_bbs_ids)
                    if os.name == 'posix':
                        move_servo(pwm_pan, a_x, 319.5, [0, 180], 1)
                        move_servo(pwm_tilt, a_y, 239.5, [30, 135], -1)

                scale_factor = display_width / img.width
                height_ratio = img.height / img.width
                img.resize((display_width, int(display_width * height_ratio)))

                img = np.asarray(img)
                if objs:
                    draw_box(img, objs, scale_factor, labels, track_bbs_ids, fps)
                    fpsmediu.append(fps)

                img = ImageTk.PhotoImage(Image.fromarray(img))

                try:
                    m1['image'] = img
                except:
                    pass
                new_win.update()
            except:
                pass
            try:
                new_win.update()
            except:
                break

    def new_win_2(self):
        global tflite_labels_name, tflite_model_name, persons_tflite, persons_labels
        fpsmediu = []
        tracking = 0
        cap = cv2.VideoCapture(0)
        if os.name == 'posix':
            pwm_pan.angle = 90
            pwm_tilt.angle = 90
        tflite_labels_name = persons_labels
        tflite_model_name = persons_tflite
        labels = read_label_file(tflite_labels_name)
        if os.name == 'posix':
            interpreter = tflite.Interpreter(tflite_model_name,
                                             experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        elif os.name == 'nt':
            interpreter = tf.lite.Interpreter(tflite_model_name)
        interpreter.allocate_tensors()

        def on_closing_a():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                cap.release()
                print("exterior, fps mediu: ", np.mean(fpsmediu))
                new_win.destroy()

        def callback(selection):
            #print(clicked.get())
            pass

        def change_state():
            if check_var.get() == 1:
                drop.configure(state='normal')
                tracking = 1
                modificare_lista(track_bbs_ids)
            else:
                drop.configure(state='disabled')
                tracking = 0

        def update_option_menu():
            global options
            menu = drop["menu"]
            menu.delete(0, "end")
            for string in options:
                menu.add_command(label=string,
                                 command=lambda value=string: clicked.set(value))
            if len(options) == 1 or clicked.get() not in options:
                clicked.set(options[0])

        def modificare_lista(tracking):
            global options
            a = ['-']
            for j in tracking:
                a.append(str(int(j[4])))
            options = a
            update_option_menu()

        new_win = Toplevel(self.parent)
        new_win.title("Exterior")
        new_win.geometry("900x720")
        new_win.configure(bg="black")
        new_win.protocol("WM_DELETE_WINDOW", on_closing_a)
        m1 = Label(new_win, bg="black")
        m1.pack(pady=15)

        check_var = IntVar()
        check_var.set(0)
        cb = Checkbutton(
            new_win,
            text="Activare/Dezactivare alegere manuala",
            variable=check_var,
            onvalue=1,
            offvalue=0,
            height=2,
            width=40,
            command=change_state
        )
        cb.pack(expand=True)

        options = ['-']
        clicked = StringVar()
        drop = OptionMenu(new_win, clicked, *options, command=callback)
        drop.configure(state='disabled')
        drop.pack(pady=30)
        display_width = 640
        fps_start_time = 0
        fps = 0
        while True:
            try:
                img = cap.read()[1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                _, scale = common.set_resized_input(
                    interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
                interpreter.invoke()
                objs = detect.get_objects(interpreter, score_threshold=0.55, image_scale=scale)
                fps_end_time = time.time()
                time_diff = fps_end_time - fps_start_time
                fps = 1 / time_diff
                fps_start_time = fps_end_time
                boxes = []
                centers = []
                for obj in objs:
                    boxes.append(np.array([obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax, obj.score]))
                    centers.append([(obj.bbox.xmin + obj.bbox.xmax) / 2, (obj.bbox.ymin + obj.bbox.ymax) / 2])
                if objs:
                    track_bbs_ids = mot_tracker.update(np.array(boxes))
                    if tracking == 1 and clicked.get() != '-':
                        id_uri = [int(tra[4]) for tra in track_bbs_ids]
                        id_urmarit = id_uri.index(int(clicked.get()))
                        a_x, a_y = get_target(track_bbs_ids, id_urmarit)
                    elif tracking == 0:
                        a_x, a_y = get_target(track_bbs_ids)
                    if os.name == 'posix':
                        move_servo(pwm_pan, a_x, 319.5, [0, 180], 1)
                        move_servo(pwm_tilt, a_y, 239.5, [30, 135], -1)

                scale_factor = display_width / img.width
                height_ratio = img.height / img.width
                img.resize((display_width, int(display_width * height_ratio)))

                img = np.asarray(img)
                if objs:
                    draw_box(img, objs, scale_factor, labels, track_bbs_ids, fps)
                    fpsmediu.append(fps)
                img = ImageTk.PhotoImage(Image.fromarray(img))
                modificare_lista(track_bbs_ids)
                try:
                    m1['image'] = img
                except:
                    pass

                new_win.update()
            except:
                pass
            try:
                new_win.update()
            except:
                break


if __name__ == "__main__":
    root = Tk()
    root.geometry('380x240')
    root.title("Alegerea contextului de functionare")
    App(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.bind('<Escape>', lambda e: root.destroy())
    root.mainloop()
