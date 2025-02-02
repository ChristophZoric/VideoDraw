#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import subprocess
import traceback

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import subprocess
import sys
import threading
import queue
import time

import utils.draw_utils as du


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def producer(annotations):
    global cnn_prediction
    global crnn_prediction
    last_annotations = None
    while True:
        time.sleep(0.5)
        with annotations_lock:
            if any(len(sublist) > 0 for sublist in annotations):
                annotations_copy = copy.deepcopy(annotations)
                if last_annotations is None or annotations_copy != last_annotations:
                    if classification_queue.empty():
                        classification_queue.put(annotations_copy)
                        print("Task added to queue:", annotations_copy)
                        last_annotations = annotations_copy
                else:
                    print("Annotations are the same, waiting for new data...")
            else:
                print("Annotations are empty, waiting for new data...")
                while not classification_queue.empty():
                    classification_queue.get()
                last_annotations = None
                cnn_prediction = "Other?"
                crnn_prediction = "Other?"


def classification_worker():
    while True:
        global cnn_prediction
        global crnn_prediction
        global cnn_time
        global crnn_time
        if not classification_queue.empty():
            annotations = classification_queue.get()
            print("Classifying annotations:", annotations)

            # Measure CNN Process Time
            start_cnn = time.time()
            cnn_process = classify(annotations, 'classification-cnn.predictor')
            stdout, stderr = cnn_process.communicate()
            end_cnn = time.time()
            cnn_time = end_cnn - start_cnn  # Time in seconds

            # Handle CNN Output
            if stdout:
                print("CNN Output:", stdout)
                cnn_prediction = " " + stdout.split('CNN Vorhersage: ')[1]
            if stderr:
                print("CNN Error:", stderr)

            # Measure CRNN Process Time
            start_crnn = time.time()
            crnn_process = classify(
                annotations, 'classification-crnn.predictor')
            stdout2, stderr2 = crnn_process.communicate()
            end_crnn = time.time()
            crnn_time = end_crnn - start_crnn  # Time in seconds

            # Handle CRNN Output
            if stdout2:
                print("CRNN Output:", stdout2)
                crnn_prediction = " " + stdout2.split('CRNN Vorhersage: ')[1]
            if stderr2:
                print("CRNN Error:", stderr2)
        else:
            time.sleep(0.1)


classification_queue = queue.Queue(maxsize=2)
annotations = [[]]
annotations_lock = threading.Lock()
worker_thread = threading.Thread(target=classification_worker, daemon=True)
producer_thread = threading.Thread(
    target=producer, args=(annotations,), daemon=True)
cnn_prediction = "Other?"
crnn_prediction = "Other?"
worker_thread.start()
producer_thread.start()
cnn_time = float('inf')
crnn_time = float('inf')


def main():
    # Argument parsing #################################################################
    args = get_args()

    # Vars ############################################################################
    global annotations
    global cnn_prediction
    global crnn_prediction
    global cnn_time
    global crnn_time
    annotationNumber = -1
    annotationStart = False
    lastDel = False

    frame_counter = 0
    frame_skip = 10

    buffer_counter = 10

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('interface/model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'interface/model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    cnn_process = None
    crnn_process = None

    while True:
        fps = cvFpsCalc.get()

        frame_counter += 1

        ## Process Key (ESC: end) #################################################
        key = cv.waitKey(5)
        if key == ord('q'):  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 0:
                    buffer_counter += 1
                    annotationStart = False

                elif hand_sign_id == 1:  # Draw gesture
                    if buffer_counter >= 0 and buffer_counter <= 5:
                        buffer_counter = -1
                        if lastDel is True:
                            annotationNumber += 1
                            annotations.append([])
                            lastDel = False
                        annotations[annotationNumber].append(
                            tuple(landmark_list[8])
                        )
                        annotationStart = True
                        continue
                    else:
                        buffer_counter = -1
                    if annotationStart is False:
                        annotationNumber += 1
                        if lastDel is True:
                            annotationNumber += 1
                            annotations.append([])
                            lastDel = False
                        annotations.append([])
                        annotationStart = True
                    annotations[annotationNumber].append(
                        tuple(landmark_list[8])
                    )

                elif hand_sign_id == 2 and annotations and frame_counter % frame_skip == 0:
                    annotations.pop()
                    annotationNumber -= 1
                    annotationStart = False
                    lastDel = True

                elif hand_sign_id == 3 and frame_counter % frame_skip == 0:
                    annotations.clear()
                    annotations.append([])
                    annotationNumber = -1
                    annotationStart = False

                else:
                    point_history.append([0, 0])
                    annotationStart = False

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                # Removed black outer Rectangle for better vision
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = du.draw_landmarks(debug_image, landmark_list)
                debug_image = du.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        # debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_annotation_history(debug_image, annotations)
        debug_image = du.draw_info(
            debug_image, fps, mode, number, cnn_prediction, crnn_prediction, cnn_time, crnn_time)

        if all(len(sublist) == 0 for sublist in annotations):
            debug_image = du.draw_instruction(debug_image)

        cv.imshow('Hand Gesture Recognition', debug_image)
    cap.release()
    cv.destroyAllWindows()


def classify(annotations, script_path):
    with open("annotations_data.txt", "w") as f:
        f.write(str(annotations))
        f.flush()
    process = subprocess.Popen(
        [sys.executable, '-m', script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process


def draw_annotation_history(image, annotations):
    for annotation in annotations:
        if len(annotation) < 2:
            continue
        for i in range(1, len(annotation)):
            start_point = annotation[i - 1]
            end_point = annotation[i]
            start_point = (int(start_point[0]), int(start_point[1]))
            end_point = (int(end_point[0]), int(end_point[1]))

            if start_point and end_point:
                cv.line(image, tuple(start_point),
                        tuple(end_point), (0, 255, 0), 2)
    return image


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'interface/model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'interface/model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


if __name__ == '__main__':
    main()
