import os
import shutil
import pandas as pd
import imageio

import config


def reset_working_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def save_data_to_csv(data, label, target_path):
    transformed_data = {}
    for landmark_id in range(len(config.ID_TO_LANDMARK)):
        for coordinate in ['x', 'y', 'z']:
            transformed_data[f'{config.ID_TO_LANDMARK[landmark_id]}_{coordinate}'] = data[landmark_id][coordinate]

    # safe data in df
    df = pd.DataFrame(transformed_data)

    compression_opts = dict(method='zip', archive_name=f'{label}.csv')
    df.to_csv(f'{target_path}/{label}.zip', index=False, compression=compression_opts)

    print(f'{label}.zip successfully saved in {target_path}!')


def create_gif(paths, class_name):
    paths = [path for path in paths if 'User1_1' in path]
    images = []
    for filename in paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{config.TARGET_PATH_GIFS}/{class_name}.gif', images)
