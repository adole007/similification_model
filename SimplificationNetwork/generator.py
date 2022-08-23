import os.path as op
import numpy as np
import ntpath
import glob
import random
import cv2
from enum import IntEnum


class DataTarget(IntEnum):
    Train = 0,
    Test = 1,
    Validate = 2,
    End = 3


class ImageIterator:
    def __init__(self, data, batch_size, img_size, column):
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.batch = self.allocate_batch()
        self.sample_count = len(data)
        self.offset = 0
        self.column = column

    def __iter__(self):
        self.offset = 0
        return self

    def __next__(self):
        batch_index = 0
        end = self.offset + self.batch_size
        while self.offset < end:
            img = cv2.imread(self.data[self.offset][self.column], cv2.IMREAD_GRAYSCALE)
            img = np.ndarray.astype(img, np.float)
            img /= 255
            self.batch[batch_index, :, :, 0] = img

            batch_index += 1
            self.offset += 1
            if self.offset == self.sample_count:
                self.offset = 0
                end = self.batch_size - batch_index

        return self.batch

    def allocate_batch(self):
        return np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 1))


def make_data_dirs(config):
    tvt = []
    base_dir = config.base_dir
    train_dir = op.join(base_dir, config.train_dir)
    val_dir = op.join(base_dir, config.validate_dir)
    test_dir = op.join(base_dir, config.test_dir)
    tvt.append([op.join(train_dir, config.image_sub_dir), op.join(train_dir, config.segmentation_sub_dir)])
    tvt.append([op.join(test_dir, config.image_sub_dir), op.join(test_dir, config.segmentation_sub_dir)])
    tvt.append([op.join(val_dir, config.image_sub_dir), op.join(val_dir, config.segmentation_sub_dir)])
    return tvt


def contrast_stretch(img, min_max):
    img -= min_max[0]
    img /= max((min_max[1] - min_max[0]), 1)
    return img


def get_min_max(img):
    min_v = np.amin(img)
    max_v = np.amax(img)
    return min_v, max_v


def min_max_contrast_stretch(img):
    return contrast_stretch(img, get_min_max(img))


STRETCH_V_PROB = 0.2
BLUR_PROB = 0.2
ADD_NOISE_PROB = 0.2
ADD_BUDGE_PROB = 0.2

MIN_MAX_RANGE = [-0.2, 0.2]
BLUR_RANGE = [1, 5]
GAUSS_VAR_RANGE = [0.15, 0.25]
SP_RATIO_RANGE = [0.4, 0.6]
SP_COVERAGE_RANGE = [0.001, 0.01]
BUDGE_RANGE = [-5, 5]


def distort_v(img):
    min_v, max_v = get_min_max(img)
    v_range = max_v - min_v
    min_v += from_range(MIN_MAX_RANGE) * v_range
    max_v += from_range(MIN_MAX_RANGE) * v_range
    if min_v > max_v:
        swap = max_v
        max_v = min_v
        min_v = swap
    return max(0, min_v), min(max_v, 1)


def round_int(f):
    return int(round(f))


def from_range(r):
    return random.uniform(r[0], r[1])


def blur(img):
    kernel_size_x = round_int(from_range(BLUR_RANGE))
    kernel_size_y = round_int(from_range(BLUR_RANGE))

    if kernel_size_x%2 == 0:
        kernel_size_x += 1

    if kernel_size_y%2 == 0:
        kernel_size_y += 1

    return cv2.GaussianBlur(img, (kernel_size_x, kernel_size_y), cv2.BORDER_DEFAULT).reshape(img.shape)


def normalise_v(img, min_max_out):
    min_max_in = get_min_max(img)
    r_out = min_max_out[1] - min_max_out[0]
    r_in = min_max_in[1] - min_max_in[0]
    img -= min_max_in[0]
    img *= r_out/r_in
    img += min_max_out[0]
    return img


def add_gauss_noise(img):
    min_max = get_min_max(img)
    row, col, ch = img.shape
    mean = 0
    var = from_range(GAUSS_VAR_RANGE)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    img = img + gauss
    return normalise_v(img, min_max).reshape(img.shape)


def add_salt_and_pepper_noise(img):
    shape = img.shape
    s_vs_p = from_range(SP_RATIO_RANGE)
    amount = from_range(SP_COVERAGE_RANGE)
    out = np.copy(img)
    out = out.reshape((shape[0], shape[1]))

    num_salt = np.ceil(amount * out.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in out.shape]
    out[tuple(coords)] = 1

    num_pepper = np.ceil(amount * out.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in out.shape]
    out[tuple(coords)] = 0
    return out.reshape(shape).reshape(img.shape)


def add_poisson_noise(img):
    min_max = get_min_max(img)
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return normalise_v(noisy, min_max).reshape(img.shape)


def add_speckle_noise(img):
    min_max = get_min_max(img)
    row, col, ch = img.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    img = img + img * gauss
    return normalise_v(img, min_max).reshape(img.shape)


NOISE_FUNC = [add_gauss_noise, add_salt_and_pepper_noise, add_poisson_noise, add_speckle_noise]


def add_noise(img):
    f_noise = NOISE_FUNC[random.randrange(0, len(NOISE_FUNC))]
    return f_noise(img)


def add_budge(img):
    min_max = get_min_max(img)
    budged = img.copy()
    x_offset = round_int(from_range(BUDGE_RANGE))
    y_offset = round_int(from_range(BUDGE_RANGE))
    src_top = max(y_offset, 0)
    src_bottom = min(img.shape[0] + y_offset, img.shape[0])
    src_left = max(x_offset, 0)
    src_right = min(img.shape[1] + x_offset, img.shape[1])
    w = src_right - src_left
    h = src_bottom - src_top
    if y_offset > 0:
        dst_top = 0
        dst_bottom = h
    else:
        dst_top = -y_offset
        dst_bottom = dst_top + h

    if x_offset > 0:
        dst_left = 0
        dst_right = w
    else:
        dst_left = -x_offset
        dst_right = dst_left + w

    img_tile = img[src_top:src_bottom, src_left:src_right]
    budge_tile = budged[dst_top:dst_bottom, dst_left:dst_right]

    budged[dst_top:dst_bottom, dst_left:dst_right] = img_tile + budge_tile
    return normalise_v(budged, min_max)


def do_test(v):
    return random.uniform(0, 1) < v


def create_img(img):
    for i in range(0, img.shape[0]):
        aug = img[i, :, :, :]
        if do_test(ADD_NOISE_PROB):
            aug = add_noise(aug)

        if do_test(ADD_BUDGE_PROB):
            aug = add_budge(aug)

        if do_test(BLUR_PROB):
            aug = blur(aug)

        if do_test(STRETCH_V_PROB):
            aug = contrast_stretch(aug, distort_v(aug))

        img[i, :, :, :] = aug

    return img


class Generator:
    def __init__(self, config):
        dirs = make_data_dirs(config)
        self.data = []
        self.img_size = config.get_input_size()
        self.batch_size = config.get_batch_size()
        for idx in range(DataTarget.Train, DataTarget.End):
            pairs = []
            imgs = glob.glob(op.join(dirs[idx][0], config.get_image_wildcard()))
            for img in imgs:
                basename = ntpath.basename(img)
                seg = op.join(dirs[idx][1], basename)
                assert(op.exists(seg))
                pairs.append([img, seg])
            self.data.append(pairs)

    def count(self, target):
        assert(DataTarget.Train <= target <= DataTarget.Validate)
        return len(self.data[target])

    def generator(self, target):
        assert(DataTarget.Train <= target <= DataTarget.Validate)
        random.shuffle(self.data[target])
        imgs = ImageIterator(self.data[target], self.batch_size, self.img_size, 0)
        segs = ImageIterator(self.data[target], self.batch_size, self.img_size, 1)
        gen = zip(imgs, segs)
        for (img, seg) in gen:
            yield (create_img(img), seg)




