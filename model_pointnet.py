import os
import glob
from timeit import default_timer as timer
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

NUM_POINTS = 16
NUM_CLASSES = 7
BATCH_SIZE = 32

tf.random.set_seed(1337)
# gets saved at C:/Users/{username}/.keras/datasets
DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

# for total runtime
start1 = timer()
folder_list = ['bathtub', 'bed', 'chair', 'night_stand', 'sofa', 'table', 'toilet']
# Create needed directories
for i in folder_list:
    if not os.path.exists('pointcloud_test_files/{}'.format(i)):
        os.makedirs('pointcloud_test_files/{}'.format(i))
    if not os.path.exists('pointcloud_train_files/{}'.format(i)):
        os.makedirs('pointcloud_train_files/{}'.format(i))


def parse_dataset(num_points, generate_point_cloud):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        pointcloud_train_files = glob.glob(os.path.join("pointcloud_train_files/{0}/*".format(os.path.basename(folder))))
        pointcloud_test_files = glob.glob(os.path.join("pointcloud_test_files/{0}/*".format(os.path.basename(folder))))

        if generate_point_cloud:
            print("generating point clouds for: {}".format(os.path.basename(folder)))
            for f in train_files:
                cloud = trimesh.load(f).sample(num_points)
                train_points.append(cloud)
                train_labels.append(i)
                np.save('pointcloud_train_files/{0}/{1}'.format(os.path.basename(folder), os.path.basename(f)), cloud)

            for f in test_files:
                cloud = trimesh.load(f).sample(num_points)
                test_points.append(cloud)
                test_labels.append(i)
                np.save('pointcloud_test_files/{0}/{1}'.format(os.path.basename(folder), os.path.basename(f)), cloud)

        else:
            for f in pointcloud_train_files:
                train_points.append(np.load(f))
                train_labels.append(i)

            for f in pointcloud_test_files:
                test_points.append(np.load(f))
                test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
    )


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def tnet(inputs, num_features):
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=regularizers.l2(1e-5),
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


train_points, test_points, train_labels, test_labels = parse_dataset(NUM_POINTS, generate_point_cloud=False)

print("Building dataset")
train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

print('shuffling test and train data')
train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

inputs = keras.Input(shape=(None, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

print("compiling model")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=["sparse_categorical_accuracy"],
              )

start2 = timer()
print("fitting model")
model.fit(train_dataset, epochs=20, validation_data=test_dataset)
end = timer()
print("fitting model took ", end - start2, ' seconds')

print("saving model")
# save model to drive
tf.keras.models.save_model(
    model=model,
    filepath='model_pointnet.h5',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None
)
print("total runtime is ", end - start1, ' seconds')
