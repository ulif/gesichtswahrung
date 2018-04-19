gesichtswahrung
===============

Toy stuff based on `face_recognition`_

Here we collect Python scripts based on `face_recognition`_, a simple, yet
awesome facial recognition API by `Adam Geitgey`_.


Preparation
-----------

To run these script, a suitable environment is required, especially the needed
Python packages and the libs they depend on need to be in place. You can create
a working runtime environment by following the steps described below.

Create a Python virtual environment::

    $ virtualenv -p /usr/bin/python3.6 py36

Activate it::

    $ source ./py36/bin/activate

Install `face_recognition`_::

    (py36) $ pip install face_recognition

Install OpenCV_ Pythoin bindings::

    (py36) $ pip install opencv-python

Now you should be able to import the `face_recognition` and `cv2` packages in
Python scripts. You can check like this::

    (py36) $ python
    >>> import face_recognition
    >>> import cv2

You can leave the interpreter pressing ``CTRL-D``. The above imports should not
raise any exception.


Running
-------

Run ``facerec_webcam`` as::

    (py36) $ python facerec_webcam

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
.. _`face_recognition`: https://github.com/ageitgey/face_recognition
.. _`OpenCV`: https://opencv.org/
