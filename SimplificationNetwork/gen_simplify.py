import glob
import random
from config import Config
from path import Path
from gen_utils import *


def main():
    config = Config("config.json")
    img_files = glob.glob(op.join(config.gen["root_dir"], config.gen["img_dir"],
                                  '*.' + config.gen["img_ext"]))
    img_size = config.gen_simplify["img_size"]
    train_test = config.gen["tvt_ratio"][0]
    val_test = config.gen["tvt_ratio"][1] + train_test
    data_dirs = make_data_dirs(config)
    file_total = len(img_files)
    for file_index, img_file in enumerate(img_files):
        print("{}/{}".format(file_index + 1, file_total))
        path_file, name = get_matching_path_file(config.gen, img_file)
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        path = Path(path_file)

        scaled_img = scale_image(img, img_size)
        simple_img = render_path(path, scaled_img, img)
        test = random.uniform(0, 1)

        img_folder, simple_folder = get_img_seg_folder(test, train_test, val_test, data_dirs)

        img_path = op.join(img_folder, name + config.img_ext)
        simple_path = op.join(simple_folder, name + config.img_ext)

        cv2.imwrite(img_path, scaled_img)
        cv2.imwrite(simple_path, simple_img)


if __name__ == "__main__":
    main()
