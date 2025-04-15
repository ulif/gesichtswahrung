gesichtswahrung
===============

Toy stuff based on `face_recognition`_, a Python wrapper around the C++ library
`dlib`_.

Here we collect Python scripts based on `face_recognition`_, a simple, yet
awesome facial recognition API by `Adam Geitgey`_.

Scripts:

     - `facerec_webcam.py`
          Capture and recognize faces live from webcam.


Preparation
-----------

To run these script, a suitable environment is required, especially the needed
Python packages and the libs they depend on need to be in place. You can create
a working runtime environment by following the steps described below.

Create a Python virtual environment::

    $ python3.11 -m venv py311

Activate it::

    $ source ./py311/bin/activate

Install `face_recognition`_::

    (py311) $ pip install face_recognition

Install OpenCV_ Pythoin bindings::

    (py311) $ pip install opencv-python

Now you should be able to import the `face_recognition` and `cv2` packages in
Python scripts. You can check like this::

    (py311) $ python
    >>> import face_recognition
    >>> import cv2

You can leave the interpreter pressing ``CTRL-D``. The above imports should not
raise any exception.

In the script `facerec_webcam.py` you can set a handfull of variables near the
top.

The ``SCREENSIZE`` setting (a tuple like ``(640, 480)``) is assumed to
represent the width and height of the fullsize screen. If set, also the
background is painted black (only) in fullscreen mode.


Running
-------

Run ``facerec_webcam`` as::

    (py311) $ python facerec_webcam.py

This script requires a working webcam connected to your device. It allows to
capture faces and to recognize these captured faces. All image data is held in
memory only. No need to create photographs beforehand.

Type ``q`` to quit. Type ``s`` to toggle mode. In `snapshot` mode the image is
frozen. You can select a face pressing `SPACE` followed by `ENTER`.


Misc
----

``Gesichtswahrung`` is a German noun meaning "Saving somebodys face", or,
literally, "Keeping of a face".


.. _`Adam Geitgey`: https://github.com/ageitgey
.. _`dlib`: https://github.com/davisking/dlib/tree/master
.. _`face_recognition`: https://github.com/ageitgey/face_recognition
.. _`OpenCV`: https://opencv.org/
