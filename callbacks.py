import tensorflow as tf
from tensorflow import keras
import resource
import gc

class MemoryCheck(keras.callbacks.Callback):
    """docstring for CustomCallback.

    for complete documentation of custom callback check out :
        https://keras.io/guides/writing_your_own_callbacks/
        https://keras.io/api/callbacks/

    'self.model' can be used to obtain parameters of the model
    e.g. 'self.model.optimizer.learning_rate'"""

    def __init__(self):
        super(MemoryCheck, self).__init__()
        self.count = 0


    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        #print("garbage collector:",gc.get_count())
        #gc.collect()
        pass

    def on_train_batch_end(self, batch, logs=None):
        #print("garbage collector:",gc.get_count())
        self.count += 1
        if self.count % 20 == 0:
            gc.collect()

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass
