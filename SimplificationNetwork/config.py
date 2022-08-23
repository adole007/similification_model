import datetime
import json
import errno
import os
import glob
import os.path as op


def create_file_directories(file_path):
    if not op.exists(op.dirname(file_path)):
        try:
            os.makedirs(op.dirname(file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class UNetLayerParams:
    def __init__(self, filters, pool_size):
        self.filters = filters
        self.pool_size = pool_size
        self.down = None
        self.up = None


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.__dict__ = json.loads(f.read())
        self.tstamp_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def _make_absolute_to_base(self, dir):
        return op.join(self.base_dir, dir)

    def get_train_dir(self):
        return self._make_absolute_to_base(self.train_dir)

    def get_test_dir(self):
        return self._make_absolute_to_base(self.test_dir)

    def get_validate_dir(self):
        return self._make_absolute_to_base(self.validate_dir)

    def get_image_subdir(self):
        return self.image_sub_dir

    def get_segmentation_subdir(self):
        return self.segmentation_sub_dir

    def make_save_dir_now(self):
        save_dir = self.save_dir
        if save_dir.startswith('/'):
            save_dir = op.join(save_dir, self.tstamp_dir + '/')
        else:
            save_dir = self._make_absolute_to_base(save_dir)
        create_file_directories(save_dir)
        return save_dir

    def get_image_wildcard(self):
        return "*{}".format(self.img_ext)

    def _count_files(self, dir, sub_dir, file_wildcard):
        wildcard = op.join(dir, sub_dir, file_wildcard)
        pattern = self._make_absolute_to_base(wildcard)
        return len(glob.glob(pattern))

    def get_train_count(self):
        return self._count_files(self.get_train_dir(), self.get_image_subdir(), self.get_image_wildcard())

    def get_test_count(self):
        return self._count_files(self.get_test_dir(), self.get_image_subdir(), self.get_image_wildcard())

    def get_validate_count(self):
        return self._count_files(self.get_validate_dir(), self.get_image_subdir(), self.get_image_wildcard())

    def get_weights_file(self):
        return self.weights_file

    def get_test_img_dir(self):
        return self._make_absolute_to_base(op.join(self.get_test_dir(), self.get_image_subdir()))

    def get_test_segmentation_dir(self):
        return self._make_absolute_to_base(op.join(self.get_test_dir(), self.get_segmentation_subdir()))

    def _get_base_log_dir(self):
        return self.base_log_dir

    def get_tensorboard_log_dir(self):
        log_dir = None
        base_log_dir = self._get_base_log_dir()
        if op.exists(base_log_dir):
            log_dir = op.join(base_log_dir, self.tstamp_dir)
        return log_dir

    def get_batch_size(self):
        return self.batch_size

    def get_target_size(self):
        size = self.target_size
        return size[0], size[1]

    def get_input_size(self):
        size = self.target_size
        return size[0], size[1], 1

    def get_layers(self):
        layers = []
        for layer in self.layers:
            layers.append(UNetLayerParams(layer[0], (layer[1], layer[2])))
        return layers
