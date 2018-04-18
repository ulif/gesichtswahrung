# -*- coding: utf-8 -*-
# facerec_webcam.py -- Detect and store faces via webcam
#
# Copyright (c) 2018 ulif
#
import face_recognition
import cv2

FONT = cv2.FONT_HERSHEY_DUPLEX
RESIZE_RATIO = 4  # for faster face recognition we shrink frames

class Faces(object):

    faces = []

    def add(self, path, name):
        image = face_recognition.load_image_file(path)
        for face in face_recognition.face_encodings(image):
            self.faces.append((face, name))
            break  # currently, we handle the 1st found face only

    def getName(self, face):
        known_faces = [x[0] for x in self.faces]
        matches = face_recognition.compare_faces(known_faces, face)
        if True in matches:
            return self.faces[matches.index(True)][1]
        return 'Unknown'

    def detect(self, frame, ratio=RESIZE_RATIO):
        """Lookup faces in `frame`.

        Return all faces found (known and unknown) with their position in frame
        and name.
        """
        # shrink frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=1.0/ratio, fy=1.0/ratio)
        # convert from BGR format (cv2) -> RGB (face_recognition)
        small_frame = small_frame[:, :, ::-1]
        locs = face_recognition.face_locations(small_frame)
        # upscale found face locations (compensate shrinking above)
        locs = [tuple([ratio * val for val in loc]) for loc in locs]
        faces_in_frame = face_recognition.face_encodings(small_frame, locs)
        for loc, face in zip(locs, faces_in_frame):
            yield loc, self.getName(face)


def draw_text_box(frame, x, y, text, scale=1.0, width=None, height=None):
    """Draw a black box in `frame` displaying `text`.

    `x`, `y` represent the *upper left* position of box to draw.
    """
    w, h = cv2.getTextSize(text, FONT, scale, 1)[0]
    w = width or w
    h = height or h
    cv2.rectangle(
        frame, (x, y), (x + w + 5, y + h + 5), (0x00, 0x00, 0x00), cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 5), (0xff, 0xff, 0xff))
    cv2.putText(
        frame, text, (x + 2, y + h + 2), FONT, scale, (0xff, 0xff, 0xff), 1)


def draw_face_box(frame, name, loc, color=(0x00, 0x00, 0xff)):
    """Draw a box on `frame` at location `loc`.

    Draws a little frame around faces on location `loc`, couloured in `color`
    (BGR model). The name `name` is put into the box bottom, like a caption.
    """
    top, right, bottom, left = loc
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(
        frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, name, (left + 6, bottom - 6), FONT, 1.0, (0xff, 0xff, 0xff), 1)


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
faces = Faces()
found_faces = []
process_this_frame = True
mode = 'DETECT'

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame and mode == 'DETECT':
        # Find all the faces and face encodings in the current frame of video
        found_faces = [x for x in faces.detect(frame)]

    process_this_frame = not process_this_frame

    # Display the results
    for loc, name in found_faces:
        draw_face_box(frame, name, loc)

    draw_text_box(frame, 0, 0, "MODE: %s" % mode)
    draw_text_box(frame, 0, 62, "<press q to quit>", 0.6)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
