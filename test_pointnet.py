import os
import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

tf.random.set_seed(1337)
DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

print('starting pointnet_model test')


def parse_test_dataset():
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))
    pointcloud_test_files = glob.glob(os.path.join("pointcloud_test_files/*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        print(i)
        # gather all files

        for f in pointcloud_test_files:
            test_points.append(np.load(f))
            test_labels.append(i)

    return (
        np.array(test_points),
        np.array(test_labels),
        class_map,
    )


test_points, test_labels, CLASS_MAP = parse_test_dataset()
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size=32)

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# load model
try:
    model = tf.keras.models.load_model('model_pointnet.h5')
except FileNotFoundError as e:
    print("model not found")
    exit(400)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_points, test_labels, batch_size=32)
print("test loss, test acc:", results)

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()


# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            os.path.basename(CLASS_MAP[preds[i].numpy()]), os.path.basename(CLASS_MAP[labels.numpy()[i]])
        )
    )
    ax.set_axis_off()
plt.show()
