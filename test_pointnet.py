import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from model_pointnet import BATCH_SIZE, NUM_POINTS, DATA_DIR


def parse_test_dataset(num_points=2048):
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        pointcloud_test_files = glob.glob(os.path.join(folder, "pointcloud_test/*"))

        for f in pointcloud_test_files:
            test_points.append(np.load(f))
            test_labels.append(i)

    return (
        np.array(test_points),
        np.array(test_labels),
        class_map,
    )


test_points, test_labels, CLASS_MAP = parse_test_dataset(NUM_POINTS)
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# load model
try:
    model = tf.keras.models.load_model('model_pointnet')
except FileNotFoundError as e:
    print("model not found")

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
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()