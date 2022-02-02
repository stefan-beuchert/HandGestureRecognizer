import cv2
import mediapipe as mp

import config


def get_coordinates(file_paths):

    # initiate empty result list
    # the output should be a list with entries for each possible landmark
    # the landmarks are a dict with the coordinates (x,y,z) and a list of values for each image
    number_of_images = len(file_paths)
    number_of_landmarks = len(config.ID_TO_LANDMARK)

    res = [None] * number_of_landmarks
    for landmark_id in range(number_of_landmarks):
        res[landmark_id] = {}
        res[landmark_id]['x'] = [None] * number_of_images
        res[landmark_id]['y'] = [None] * number_of_images
        res[landmark_id]['z'] = [None] * number_of_images

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        for frame_count, file in enumerate(file_paths):

            # Read an image
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # check if result is valid
            if not results.multi_hand_landmarks:
                print('no hand detected')
                continue
            elif len(results.multi_hand_landmarks) > 1:
                print('more then one hand detected, not processing the image!')
                continue

            for landmark_id, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                res[landmark_id]['x'][frame_count] = landmark.x
                res[landmark_id]['y'][frame_count] = landmark.y
                res[landmark_id]['z'][frame_count] = landmark.z

                # TODO z coordinate is not normalized -> x and y are ranged between 0 and 1,
                #  but z is dependent on its distance to the palm -> should be normalized for VY

    return res


def get_coordinates_for_one_image(image):

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        # check if result is valid
        if not results.multi_hand_landmarks:
            #print('no hand detected')
            return None
        elif len(results.multi_hand_landmarks) > 1:
            print('more then one hand detected, not processing the image!')
            return None

        list_of_x_coordinates = []
        list_of_y_coordinates = []
        list_of_z_coordinates = []

        for landmark_id, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            list_of_x_coordinates.append(landmark.x)
            list_of_y_coordinates.append(landmark.y)
            list_of_z_coordinates.append(landmark.z)

    return (list_of_x_coordinates, list_of_y_coordinates, list_of_z_coordinates)
