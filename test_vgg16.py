import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gen_circle import synthetic_gen, oval_gen
import os

# for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['OPENBLAS_NUM_THREADS'] = '4'
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)


# given image and a label, plots the image + rectangle
def plot_pred(img, p):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = Rectangle(xy=(p[1] * 128, p[0] * 128), width=p[2] * 128, height=p[2] * 128, linewidth=1, edgecolor='g',
                     facecolor='none')
    ax.add_patch(rect)
    plt.show()


# load model
try:
    new_model = tf.keras.models.load_model('model')
except FileNotFoundError as e:
    print("model not found")

# generate new image
x, _ = next(synthetic_gen())
# predict
prediction = new_model.predict(x)

# check with oval instead
x1, _ = oval_gen()
prediction_oval = new_model.predict(x1)

# examine 1 image
im = x[0]
p = prediction[0]
plot_pred(im, p)
