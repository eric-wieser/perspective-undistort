import numpy as np
from numpy import sin, cos
import cv2

def rot_x(x):
    return np.array([
        [     1, 0,       0,      0],
        [     0, cos(x), -sin(x), 0],
        [     0, sin(x),  cos(x), 0],
        [     0, 0,       0,      1]
    ])
def rot_y(y):
    return np.array([
        [cos(y), 0, -sin(y), 0],
        [     0, 1,       0, 0],
        [sin(y), 0,  cos(y), 0],
        [     0, 0,       0, 1]
    ])
def rot_z(z):
    return np.array([
        [ cos(z), sin(z), 0, 0],
        [-sin(z), cos(z), 0, 0],
        [      0,      0, 1, 0],
        [      0,      0, 0, 1]
    ])

def translate(x=0, y=0, z=0):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

def scale(x=1, y=1, z=1, all=None):
    if all is not None:
        x = y = z = all

    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ])
from contextlib import contextmanager

def find_camera():
    for i in range(3):
        capture = cv2.VideoCapture(i)
        if capture:
            for i in range(2):
                res, data = capture.read()
                if res:
                    return capture
            capture.release()
    raise EnvironmentError("no camera")

@contextmanager
def get_camera():
    capture = find_camera()

    try:
        # dummy read required
        capture.read()
        yield capture
    finally:
        capture.release()

def get_acc():
    import ctypes
    'Sensorsapi.dll'