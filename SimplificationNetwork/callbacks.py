import tensorflow.python.keras as keras


def create_callbacks(config, weights_file, custom_callbacks=[]):
    callbacks = [keras.callbacks.ModelCheckpoint(weights_file, monitor='loss', verbose=1, save_best_only=True)]
    log_dir = config.get_tensorboard_log_dir()
    if log_dir is not None:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, update_freq='batch', profile_batch=0))
    if custom_callbacks and len(custom_callbacks) > 0:
        callbacks.extend(custom_callbacks)
    return callbacks
