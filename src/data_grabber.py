import os


class Cursor:
    def __init__(self, data_path, file_ending):

        # get all target files in target directory
        target_files_in_target_directory = []

        for file in os.listdir(data_path):
            if file.endswith('.' + file_ending):
                target_files_in_target_directory.append(os.path.join(data_path, file))

        number_of_available_files = len(target_files_in_target_directory)

        if number_of_available_files < 1:
            raise ValueError(f'cursor initiated with zero {file_ending} elements in target directors {data_path} '
                             f'please check if this is the correct path to your data')

        self.__number_of_files = number_of_available_files
        self.__target_files = target_files_in_target_directory
        self.__cursor_position = 0  # set position to first index of self.target_folders
        self.more_files_available = True

    def get_file(self):
        return self.__target_files[self.__cursor_position]

    def move_to_next_file(self):
        if not self.more_files_available:
            raise ValueError('no more files available. Cursor reached end of possible files')

        self.__cursor_position += 1

        if self.__cursor_position + 1 == self.__number_of_files:
            self.more_files_available = False


def get_image_paths(path):
    # check for folders
    folder_content = os.listdir(path)

    # check that initial folder only has one sub folder
    number_of_elements = len(folder_content)
    if number_of_elements != 1:
        if number_of_elements < 1:
            raise Exception('No folders fount that could contain images')
        if number_of_elements > 1:
            raise Exception('More then one folder found, but should be exactly one!')

    label = folder_content[0]

    # get all paths to .png files in the folder
    list_of_image_paths = []

    for root, dirs, files in os.walk(f'{path}/{label}', topdown=True):
        for name in files:
            list_of_image_paths.append(os.path.join(root, name))

    return label, list_of_image_paths
