#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# facerec_webcam.py -- Detect and store faces via webcam
#
# Copyright (c) 2018 ulif
#
# This software may be modified and distributed under the terms of the MIT
# licence.  See the LICENSE file for details.
#
import face_recognition
import cv2
import random
import logging
import sys
import numpy as np

logger = logging.getLogger("ulif.facerec_webcam")

FONT = cv2.FONT_HERSHEY_DUPLEX
RESIZE_RATIO = 4   # for faster face recognition we shrink frames
MAX_FACES = 10     # maximum number of faces we look for (reduce load)
VIDEO_SRC = 0      # the number of video source we want to use (first one: 0)
DEFAULT_NAME = 'Unknown'
SUSPECT_NAMES = ["Drama-Queen", "Prinz.essin", "Held.in", "Kurt", "Jaqueline", "Weltretter.in"]
FULLSCREEN = True # Initial window state
SCREENSIZE = None  # (width, height) of fullscreen resolution
# SCREENSIZE = (1920, 1080)  # Enables fully stretched fullscreen and black background if set


class Faces(object):

    faces = []
    current_num = 0

    def add(self, face):
        """Register `face`.

        `face` is expected to be an encoded face, as returned by
        `face_recognition.face_encodings()`.

        We store at most ``MAX_FACES`` for recognition in order to reduce load
        (more faces to compare, more time needed to compute a frame).
        """
        self.current_num += 1
        name = 'Suspect #%s' % self.current_num
        name = random.choice(SUSPECT_NAMES)
        self.faces.append((face, name))
        self.faces = self.faces[-MAX_FACES:]

    def addFromImage(self, path, name):
        """Add first image from image file located in `path`.
        """
        image = face_recognition.load_image_file(path)
        for face in face_recognition.face_encodings(image):
            self.faces.append((face, name))
            break  # currently, we handle the 1st found face only

    def getName(self, face):
        known_faces = [x[0] for x in self.faces]
        matches = face_recognition.compare_faces(known_faces, face)
        if True in matches:
            return self.faces[matches.index(True)][1]
        return DEFAULT_NAME

    def detect(self, frame, ratio=RESIZE_RATIO):
        """Lookup faces in `frame`.

        Return all faces found (known and unknown) with as tuples (position in
        frame, name, encoding). The position is provided as (top, right,
        bottom, left), the name is a string and encoding is an numpy array
        containing the "encoding" of a face.
        """
        # shrink frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=1.0/ratio, fy=1.0/ratio)
        # convert from BGR format (cv2) -> RGB (face_recognition)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(small_frame)
        faces_in_frame = face_recognition.face_encodings(small_frame, locs)
        # upscale found face locations (compensate shrinking above)
        locs = [tuple([ratio * val for val in loc]) for loc in locs]
        for loc, face in zip(locs, faces_in_frame):
            yield loc, self.getName(face), face


def draw_text_box(frame, x, y, text, scale=1.0, width=None, height=None):
    """Draw a black box in `frame` displaying `text`.

    `x`, `y` represent the *upper left* position of box to draw.
    """
    w, h = cv2.getTextSize(text, FONT, scale, 1)[0]
    w = width or w
    h = height or h
    cv2.rectangle(
        frame, (x, y), (x + w + 5, y + h + 5), (0x00, 0x00, 0x00), cv2.FILLED)
    cv2.putText(
        frame, text, (x + 2, y + h + 2), FONT, scale, (0xff, 0xff, 0xff), 1)


def draw_face_box(frame, name, loc, color=(0x00, 0x00, 0xff)):
    """Draw a box on `frame` at location `loc`.

    Draws a little frame around faces on location `loc`, coloured in `color`
    (BGR model). The name `name` is put into the box bottom, like a caption.
    """
    top, right, bottom, left = loc
    fg = (0xff, 0xff, 0xff)
    if sum(color) > 0x180:
        fg = (0x00, 0x00, 0x00)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(
        frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, name, (left + 6, bottom - 6), FONT, 1.0, fg, 1)


def toggle_mode(mode):
    if mode == 'DETECT':
        mode = 'SNAPSHOT'
        picked = 0
    else:
        mode = 'DETECT'
        picked = None
    return mode, picked


def toggle_fullscreen(fullscreen):
    new_val = cv2.WINDOW_FULLSCREEN
    if fullscreen:
        new_val = cv2.WINDOW_NORMAL
    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, new_val)
    return not fullscreen


def prepare_fullscreen(cam_format, screen_size):
    """Compute offsets and scale-up factors for fullscreen.

    Also create a black background image, that can be merged with the actual
    cam images to provide black margins if the ratio of cam and screen are
    different.

    Returns a factor for stretching cam format so that it fits into given
    screen size (keeping the cam ratio), offsets to display the stretched cam
    images in the middle of the screen and a black image of screen size.
    """
    screen_w, screen_h = screen_size
    frame_w, frame_h = cam_format
    scale_up = min([(screen_w / frame_w), (screen_h / frame_h)])
    bg_img = np.zeros((screen_h, screen_w, 3), np.uint8)
    offset_x, offset_y = (
            int((screen_w - scale_up * frame_w) / 2),
            int((screen_h - scale_up * frame_h) / 2))
    return scale_up, offset_x, offset_y, bg_img


def draw_modestate(frame, mode):
    draw_text_box(frame, 0, 0, "MODE: %s" % mode)
    bottom = frame.shape[0]
    draw_text_box(
        frame, 0, bottom - 25, "<press q to quit, s to toggle mode>", 0.8)
    if mode == 'SNAPSHOT':
        draw_text_box(frame, 0, 27, 'SPC to select, ENTER to choose', 0.8)


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(VIDEO_SRC)


# Initialize some variables
faces = Faces()
found_faces = []
process_this_frame = True
mode = 'DETECT'
picked_face = None
fullscreen_mode = FULLSCREEN
cam_format = (
    video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
    video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

if SCREENSIZE:
    scale_up, offset_x, offset_y, bg_image = prepare_fullscreen(
            cam_format, SCREENSIZE)
else:
    logger.warning("No SCREENSIZE set. Fullscreen mode might not fill complete screen and render background white or grey")

logger.info("CAM format: %s" % (cam_format, ))

toggle_fullscreen(not fullscreen_mode)
while video_capture.isOpened():
    # Grab a single frame of video
    if mode == 'DETECT':
        ret, frame = video_capture.read()
        if frame is None:
            logger.fatal("Not a valid video capture source: %s" % VIDEO_SRC)
            sys.exit(1)
        if ret is not True:
            sys.exit(1)

    # Stretch image to fit into fullscreen (also paint background black)
    if fullscreen_mode and SCREENSIZE and mode == 'DETECT':
        frame = cv2.resize(
            frame, dsize=None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_LINEAR)
        background = bg_image.copy()
        background[offset_y:offset_y + frame.shape[0], offset_x:offset_x + frame.shape[1]] = frame
        frame = background

    # Only process every other frame of video to save time
    if process_this_frame and mode == 'DETECT':
        # Find all the faces and face encodings in the current frame of video
        found_faces = [x for x in faces.detect(frame)]

    process_this_frame = not process_this_frame

    # Display the results
    for num, face in enumerate(found_faces):
        loc, name, enc = face
        color = (0x00, 0x00, 0xff)
        if picked_face == num:
            color = (0x00, 0xff, 0xff)
        draw_face_box(frame, name, loc, color=color)

    draw_modestate(frame, mode)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
    elif key == ord('s'):
        mode, picked_face = toggle_mode(mode)
    elif key == ord(' '):
        if mode == 'SNAPSHOT' and len(found_faces):
            picked_face = (picked_face + 1) % len(found_faces)
    elif key == ord('f'):
        fullscreen_mode = toggle_fullscreen(fullscreen_mode)
    elif key == 13 and mode == 'SNAPSHOT' and len(found_faces):
        faces.add(found_faces[picked_face][2])
        mode, picked_face = toggle_mode(mode)


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
