import numpy as np
import matplotlib.pyplot as plt


def synthetic_gen(batch_size=64):
    # enable generating infinite amount of batches
    while True:
        # generate black images in the wanted size
        X = np.zeros((batch_size, 128, 128, 3))
        Y = np.zeros((batch_size, 3))
        # fill each image
        for i in range(batch_size):
            x = np.random.randint(8, 120)
            y = np.random.randint(8, 120)
            a = min(128 - max(x, y), min(x, y))
            r = np.random.randint(4, a)
            for x_i in range(128):
                for y_i in range(128):
                    if ((x_i - x) ** 2) + ((y_i - y) ** 2) < r ** 2:
                        X[i, x_i, y_i, :] = 1
            Y[i, 0] = (x - r) / 128.
            Y[i, 1] = (y - r) / 128.
            Y[i, 2] = 2 * r / 128.
        yield X, Y


def oval_gen(batch_size=1):
    # generate black images in the wanted size
    X = np.zeros((batch_size, 128, 128, 3))
    Y = np.zeros((batch_size, 3))

    # fill each image
    x = np.random.randint(8, 120)
    y = np.random.randint(8, 120)
    a = min(128 - max(x, y), min(x, y))
    r = np.random.randint(4, a)
    for x_i in range(128):
        for y_i in range(128):
            if ((x_i - x) ** 2) + ((y_i - y) ** 4) < r ** 2:
                X[0, x_i, y_i, :] = 1
    Y[0, 0] = (x - r) / 128.
    Y[0, 1] = (y - r) / 128.
    Y[0, 2] = 2 * r / 128.

    return X, Y


# sanity check - plot the images
#x, y = next(synthetic_gen())
x, y = oval_gen()
plt.imshow(x[0])
