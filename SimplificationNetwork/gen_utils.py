import os
import os.path as op
import ntpath
import numpy as np
import cv2
import errno


def get_matching_path_file(config, image_file_path):
    basename = ntpath.basename(image_file_path)
    name = basename.replace(config["img_ext"], config["path_ext"])
    return op.join(config["root_dir"], config["path_dir"], name), op.splitext(basename)[0]


def scale_image_for_tiling(img, tile_count, tile_size):
    width = tile_count[0] * tile_size[0]
    height = tile_count[1] * tile_size[1]
    return cv2.resize(img, (width, height))


def scale_image(img, img_size):
    return cv2.resize(img, (img_size[0], img_size[1]))


def render_path(path, scaled_img, img):
    path_img = np.full(scaled_img.shape, 255, dtype=np.uint8)
    strokes = path.get_scaled_path([scaled_img.shape[1]/img.shape[1], scaled_img.shape[0]/img.shape[0]])
    for stroke in strokes:
        points = np.array([[int(round(point[0])), int(round(point[1]))] for point in stroke])
        cv2.polylines(path_img, [points], False, 0, 1)
    return path_img


def make_data_dirs(config):
    tvt = []
    base_dir = config.base_dir
    train_dir = op.join(base_dir, config.train_dir)
    val_dir = op.join(base_dir, config.validate_dir)
    test_dir = op.join(base_dir, config.test_dir)
    tvt.append([op.join(train_dir, config.image_sub_dir), op.join(train_dir, config.segmentation_sub_dir)])
    tvt.append([op.join(val_dir, config.image_sub_dir), op.join(val_dir, config.segmentation_sub_dir)])
    tvt.append([op.join(test_dir, config.image_sub_dir), op.join(test_dir, config.segmentation_sub_dir)])
    for type in tvt:
        if not op.exists(type[0]):
            os.makedirs(type[0])
        if not op.exists(type[1]):
            os.makedirs(type[1])
    return tvt


def make_data_file_path(name, folder, ext, layout_index, tile_x, tile_y):
    return op.join(folder, "{}_{}_{}_{}{}".format(name, layout_index, tile_x, tile_y, ext))


def get_img_seg_folder(test, train_test, val_test, data_dirs):
    data_type = 2
    if test < train_test:
        data_type = 0
    elif test < val_test:
        data_type = 1

    return data_dirs[data_type][0], data_dirs[data_type][1]


def get_img_seg_paths(test, train_test, val_test, img_ext, name, data_dirs, layout_index, tile_x, tile_y):
    img_folder, seg_folder = get_img_seg_folder(test, train_test, val_test, data_dirs)

    img_path = make_data_file_path(name, img_folder, img_ext, layout_index, tile_x, tile_y)
    seg_path = make_data_file_path(name, seg_folder, img_ext, layout_index, tile_x, tile_y)
    return img_path, seg_path


def create_file_directories(file_path):
    if not os.path.exists(op.dirname(file_path)):
        try:
            os.makedirs(op.dirname(file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

