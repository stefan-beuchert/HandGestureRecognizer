import os
import shutil
import pandas as pd

import config


def reset_working_directory(path):
    # create empty folder for unzipped data
    path_for_unzipped_data = path
    if os.path.exists(path_for_unzipped_data):
        shutil.rmtree(path_for_unzipped_data)
    os.mkdir(path_for_unzipped_data)


def save_data_to_csv(data, label, target_path):
    transformed_data = {}
    for landmark_id in range(len(config.ID_TO_LANDMARK)):
        for coordinate in ['x', 'y', 'z']:
            transformed_data[f'{config.ID_TO_LANDMARK[landmark_id]}_{coordinate}'] = data[landmark_id][coordinate]

    # safe data in df
    df = pd.DataFrame(transformed_data)

    compression_opts = dict(method='zip', archive_name=f'{label}.csv')
    df.to_csv(f'{target_path}/{label}.zip', index=False, compression=compression_opts)

    print(f'{label}.csv successfully saved in {target_path}!')





