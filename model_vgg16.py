import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from gen_circle import synthetic_gen

BATCH_SIZE = 16
EPOCH_SIZE = 16

# transfer learning - load pre-trained vgg and replace its head
vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
x = Flatten()(vgg.output)
x = Dense(3, activation='sigmoid')(x)
model1 = Model(vgg.input, x)
model1.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))

# needs steps per epoch since the generator is infinite
model1.fit(synthetic_gen(batch_size=BATCH_SIZE), steps_per_epoch=EPOCH_SIZE, epochs=25)
model1.save('model_vgg16.h5')



