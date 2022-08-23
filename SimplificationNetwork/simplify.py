from gen_utils import create_file_directories
import os
import os.path as op
from model_simplify import simplify
from generator import Generator, DataTarget
from shutil import copyfile
import tensorflow as tf
import tensorflow.python.keras.callbacks as callbacks

from config import Config
from callbacks import create_callbacks

CONFIG_FILE = "config.json"


class SaveWeightsCallback(callbacks.Callback):
    def __init__(self, config, save_dir, weights_file):
        super(SaveWeightsCallback, self).__init__()
        self.config = config
        self.save_dir = save_dir
        self.weights_file = weights_file

    def on_epoch_end(self, epoch, logs=None):
        epoch_dir = os.path.join(self.save_dir, str(epoch) + "/")
        create_file_directories(epoch_dir)

        copyfile(self.weights_file, os.path.join(epoch_dir, self.config.get_weights_file()))


def rounded(n):
    return int(round(n))


def main():
    print("Simplifier neural network trainer. Using GPU: {}. TF Version: {}".format(tf.test.is_gpu_available(),
                                                                                    tf.__version__))

    config = Config(CONFIG_FILE)
    batch_size = config.get_batch_size()

    gen = Generator(config)

    num_train = gen.count(DataTarget.Train)
    num_val = gen.count(DataTarget.Validate)

    save_dir = config.make_save_dir_now()

    plot_file = os.path.join(save_dir, 'out.plot')
    if op.exists(plot_file):
        os.remove(plot_file)

    weights_file = config.get_weights_file()
    log_format = '{:.2f} {} {:.6f} {:.6f} {:.6f} {:.6f}\n'

    target_size = config.get_target_size()
    if os.path.exists(weights_file):
        model = simplify(pretrained_weights=weights_file, input_size=(target_size[1], target_size[0], 1))
    else:
        model = simplify(input_size=(target_size[1], target_size[0], 1))

    callbacks = create_callbacks(config, weights_file, [SaveWeightsCallback(config, save_dir, weights_file)])

    model.fit_generator(gen.generator(DataTarget.Train),
                        steps_per_epoch=rounded(num_train / batch_size),
                        epochs=200,
                        initial_epoch=0,
                        callbacks=callbacks,
                        validation_data=gen.generator(DataTarget.Validate),
                        validation_steps=rounded(num_val / batch_size))


if __name__ == '__main__':
    main()
