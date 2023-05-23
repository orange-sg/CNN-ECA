import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, Dropout, Reshape, Conv1D, Activation
from tensorflow.keras.layers import Flatten, InputLayer, GlobalAveragePooling2D, Multiply

# This code is used to construct a CNN_ECA model, which is based on the CNN model and the ECA module.
# The CNN model is used to extract spatial features, and the ECA module is used to enhance the feature expression ability of CNN.
# The input of CNN_ECA model is the data of 8 variables in 46*71 grid, and the output is the data of 464 stations.
class CNN_ECA(Model):

    def __init__(self, k=3, channel=8, latn=46, lonn=71, object_staionn=464, use_eca_block=True):
        super(CNN_ECA, self).__init__()
        self.inp = InputLayer(input_shape=(channel, latn, lonn))
        self.c1 = Conv2D(filters=64, kernel_size=k, activation='relu', padding='same')
        self.b1 = BatchNormalization()
        self.d1 = Dropout(0.1)
        self.c2 = Conv2D(filters=32, kernel_size=k, activation='relu', padding='same')
        self.b2 = BatchNormalization()
        self.d2 = Dropout(0.2)
        self.c3 = Conv2D(filters=10, kernel_size=k, activation='relu', padding='same')
        self.b3 = BatchNormalization()
        self.d3 = Dropout(0.2)
        self.eca = use_eca_block
        self.ga = GlobalAveragePooling2D(data_format='channels_first')
        self.re = Reshape((-1, channel, 1))
        self.con = Conv1D(1,kernel_size=k, padding='same')
        self.ac = Activation('sigmoid')
        self.mu = Multiply()
        self.ff = Flatten()
        self.f1 = Dense(object_staionn, activation='linear', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self,input):
        x = input
        out = self.inp(x)
        out = self.c1(out)
        out = self.b1(out)
        out = self.d1(out)
        out = self.c2(out)
        out = self.b2(out)
        out = self.d2(out)
        out = self.c3(out)
        out = self.b3(out)
        out = self.d3(out)
        if self.eca:
            eca_input = out
            out = self.ga(out)
            out = self.re(out)
            out = self.con(out)
            out = self.ac(out)
            out = tf.expand_dims(out, -1)
            out = self.mu([eca_input, out])
        out = self.ff(out)
        out = self.f1(out)
        return out

model = CNN_ECA()