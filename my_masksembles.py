import tensorflow_model_optimization as tfmot
from masksembles.keras import Masksembles2D, Masksembles1D


class MyMasksembles2D(Masksembles2D, tfmot.sparsity.keras.PrunableLayer):

    def __init__(self, n, scale):
        super(MyMasksembles2D, self).__init__(n, scale)

    def get_prunable_weights(self):
        return [self.masks]


class MyMasksembles1D(Masksembles1D, tfmot.sparsity.keras.PrunableLayer):

    def get_prunable_weights(self):
        #
        return [self.masks]
