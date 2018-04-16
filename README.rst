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


Misc
----

``Gesichtswahrung`` is a German noun meaning "Saving somebodys face", or,
literally, "Keeping of a face".


.. _`Adam Geitgey`: https://github.com/ageitgey
.. _`face_recognition`: https://github.com/ageitgey/face_recognition
.. _`OpenCV`: https://opencv.org/
