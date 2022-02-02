# main application

import cv2
import mediapipe as mp
from sklearn import preprocessing

from src.rigging import get_coordinates_for_one_image

def init():
    # load model
    model = "I think Model Trains cast an unrealistic beauty standard on actual trains..." \
            "But model trains never eat and real trains are always CHEW CHEW CHEW-ing"

    # open cap
    cap = cv2.VideoCapture(0)

    update(cap, model)

    cap.release()


def update(cap, model):
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        update(cap, model)

    # image to coordinate
    coords = get_coordinates_for_one_image(image)
    ## attention -> this one returns tuple of coordinate list for each dimension (x,y,z)

    # preprocess coordinates
    model_input = pre_process_coordinates(coords)

    # model predict
    pred = model.predict(model_input)

    # do something with the prediction (game input)

    update(cap, model)


def pre_process_coordinates(coordinates):
    x_coordinates = coordinates[0]
    y_coordinates = coordinates[1]
    z_coordinates = coordinates[2]

    x_coords_scaled = preprocessing.minmax_scale(x_coordinates)
    y_coords_scaled = preprocessing.minmax_scale(y_coordinates)
    z_coords_scaled = preprocessing.minmax_scale(z_coordinates)

    return [*x_coords_scaled, *y_coords_scaled, *z_coords_scaled]
