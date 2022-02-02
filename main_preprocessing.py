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
import pandas as pd
import os
from sklearn import preprocessing

import config
from src.data_grabber import Cursor, get_image_paths
from src.helper import reset_working_directory, save_data_to_csv, create_gif
from src.rigging import get_coordinates


def png_to_csv():
    cursor = Cursor(config.SOURCE_PATH, 'zip')

    reset_working_directory(config.TARGET_PATH)
    reset_working_directory(config.TARGET_PATH_GIFS)
    reset_working_directory(config.PLAYGROUND_PATH)

    file_available = True
    while file_available:

        file = cursor.get_file()

        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(config.PLAYGROUND_PATH)

        # get list of images for this class
        label, image_paths = get_image_paths(config.PLAYGROUND_PATH)

        # create_and_save_on_example_gif_for_each_class
        create_gif(image_paths, label)

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


def data_preprocessing():
    # unzipped csv data per class to playground folder
    cursor_for_zipped_files = Cursor(config.TARGET_PATH, 'zip')
    reset_working_directory(config.PLAYGROUND_PATH)

    # iterate through zipped files
    file_available = True
    while file_available:
        file = cursor_for_zipped_files.get_file()

        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(config.PLAYGROUND_PATH)

        if not cursor_for_zipped_files.more_files_available:
            file_available = False
        else:
            cursor_for_zipped_files.move_to_next_file()

    # load each file
    frames = []

    cursor_for_csv_files = Cursor(config.PLAYGROUND_PATH, 'csv')
    file_available = True
    while file_available:
        file = cursor_for_csv_files.get_file()

        class_data_frame = pd.read_csv(file)

        # delete emtpy rows
        size_with_all_rows = len(class_data_frame)
        class_data_frame = class_data_frame.dropna()
        size_after_dropping_nan_rows = len(class_data_frame)

        print(f'From {size_with_all_rows} rows '
              f'{size_with_all_rows - size_after_dropping_nan_rows} have been dropped due to NaN values')

        # normalize row-wise ... this zooms onto hand (x and y between 0 and 1), but stretches z in proportionally

        # scale row between 0 and 1 (for x, y, z separately)
        x_data = class_data_frame.filter(regex='x$', axis=1)
        x_data_col_names = x_data.columns
        x_data_numpy_array = preprocessing.minmax_scale(x_data.T).T

        y_data = class_data_frame.filter(regex='y$', axis=1)
        y_data_col_names = y_data.columns
        y_data_numpy_array = preprocessing.minmax_scale(y_data.T).T

        z_data = class_data_frame.filter(regex='z$', axis=1)
        z_data_col_names = z_data.columns
        z_data_numpy_array = preprocessing.minmax_scale(z_data.T).T

        all_data = [None] * len(x_data)
        for frame_counter in range(len(x_data)):
            all_data[frame_counter] = [*x_data_numpy_array[frame_counter],
                                       *y_data_numpy_array[frame_counter],
                                       *z_data_numpy_array[frame_counter]]

        all_cols = [*x_data_col_names, *y_data_col_names, *z_data_col_names]

        class_data_frame = pd.DataFrame(data=all_data, columns=all_cols)

        # add column with class label
        filename = os.path.splitext(os.path.basename(file))[0]
        class_data_frame['label'] = config.CLASS_ID_TO_LABEL[filename]

        # append df to big result list
        frames.append(class_data_frame)

        if not cursor_for_csv_files.more_files_available:
            file_available = False
        else:
            cursor_for_csv_files.move_to_next_file()

    # create on big df
    result = pd.concat(frames)

    # save data to big csv
    compression_opts = dict(method='zip', archive_name=f'{config.FINAL_OUTPUT_NAME}.csv')
    result.to_csv(f'{config.TARGET_PATH}/{config.FINAL_OUTPUT_NAME}.zip', index=False, compression=compression_opts)

    print(f'{config.FINAL_OUTPUT_NAME}.zip successfully saved in {config.TARGET_PATH}!')

    # clean working space
    reset_working_directory(config.PLAYGROUND_PATH)


if True:
    # load raw data, convert images to coordinates, save as csv per class
    png_to_csv()

    # load class csvs, preprocess and save as single csv over all classes
    data_preprocessing()

    print('PreProcessing done!')
