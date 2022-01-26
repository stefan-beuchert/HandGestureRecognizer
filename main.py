# get name of files in data foulder
# for file in fieles:
# 1 unzippen
# 2 go through each picture in subfoulders
# 3 scale on of them down and create a gif
# 4 convert image to coordinates with google framework
# 5 save in csv mit user_id, gesture_name, framej_id ...
# 6 delete unzipped file

# for each csv
# look at outliers
# check which gestures can be used for single frame interpretation (look at gifs)
# add all csvs to one big file?
import zipfile

import config
from src.data_grabber import Cursor, get_image_paths
from src.helper import reset_working_directory, save_data_to_csv
from src.mediapipe import get_coordinates


def png_to_csv():
    cursor = Cursor(config.SOURCE_PATH, 'zip')

    reset_working_directory(config.TARGET_PATH)
    reset_working_directory(config.PLAYGROUND_PATH)

    file_available = True
    while file_available:

        file = cursor.get_file()

        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(config.PLAYGROUND_PATH)

        # get list of images for this class
        label, image_paths = get_image_paths(config.PLAYGROUND_PATH)

        # images to coordinates
        coordinates = get_coordinates(image_paths)

        # save coordinates to csv file
        save_data_to_csv(coordinates, label, config.TARGET_PATH)

        # delete the unzipped file
        reset_working_directory(config.PLAYGROUND_PATH)

        if not cursor.more_files_available:
            file_available = False
        else:
            cursor.move_to_next_file()


def preprocessing():
    # load data
    # delete empty rows
    # normalize z columns

    pass


if __name__ == '__main__':
    # load raw data, convert images to coordinates, save as csv per class
    png_to_csv()

    # load class csvs, preprocess and save as single csv over all classes


    print('done')
