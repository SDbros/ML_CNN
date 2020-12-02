import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from matplotlib.patches import Rectangle
from gen_circle import synthetic_gen

BATCH_SIZE = 64
EPOCH_SIZE = 64

# transfer learning - load pre-trained vgg and replace its head
vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
x = Flatten()(vgg.output)
x = Dense(3, activation='sigmoid')(x)
model1 = Model(vgg.input, x)
model1.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
# plot the model
#plot_model(model1, "first_model.png", show_shapes=True, expand_nested=False)
# needs steps per epoch since the generator is infinite
model1.fit_generator(synthetic_gen(), steps_per_epoch=EPOCH_SIZE, epochs=5)


# given image and a label, plots the image + rectangle
def plot_pred(img, p):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = Rectangle(xy=(p[1] * 128, p[0] * 128), width=p[2] * 128, height=p[2] * 128, linewidth=1, edgecolor='g',
                     facecolor='none')
    ax.add_patch(rect)
    plt.show()


# generate new image
x, _ = next(synthetic_gen())
# predict
pred = model1.predict(x)
# examine 1 image
im = x[0]
p = pred[0]
plot_pred(im, p)
