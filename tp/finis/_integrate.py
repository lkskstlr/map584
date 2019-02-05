import numpy as np


def form_int_1d(order):
    if order == 1:
        x = np.array([1 / 2])
        w = np.array([1.0])

        return x, w

    if (order == 2) or (order == 3):
        x = np.array([1 / 2 - 1 / np.sqrt(12), 1 / 2 + 1 / np.sqrt(12)])
        w = np.array([1 / 2, 1 / 2])

        return x, w

    if (order == 4) or (order == 5):
        x = np.array([1 / 2 - 1 / 2 * np.sqrt(3 / 5), 1 /
                      2, 1 / 2 + 1 / 2 * np.sqrt(3 / 5)])
        w = np.array([5 / 18, 8 / 18, 5 / 18])

        return x, w

    raise NotImplementedError("Only orders 1, ..., 5 are implemented")


def form_int_2d(order):
    if order == 1:
        x = np.array([1 / 3])
        y = np.array([1 / 3])
        w = np.array([1 / 2])

        return x, y, w

    if order == 2:
        x = np.array([1 / 6, 2 / 3, 1 / 6])
        y = np.array([1 / 6, 1 / 6, 2 / 3])
        w = np.array([1 / 6, 1 / 6, 1 / 6])

        return x, y, w

    if order == 3:
        x = np.array([1 / 3, 1 / 5, 3 / 5, 1 / 5])
        y = np.array([1 / 3, 1 / 5, 1 / 5, 3 / 5])
        w = np.array([-9 / 32, 25 / 96, 25 / 96, 25 / 96])

        return x, y, w

    if order == 4:
        x = np.array([1 / 2, 1 / 2, 0, 1 / 6, 1 / 6, 2 / 3])
        y = np.array([1 / 2, 0, 1 / 2, 1 / 6, 2 / 3, 1 / 6])
        w = np.array([1 / 60, 1 / 60, 1 / 60, 3 / 20, 3 / 20, 3 / 20])

        return x, y, w

    if (order == 5) or (order == 6):
        a = (6 + np.sqrt(15)) / 21
        b = (6 - np.sqrt(15)) / 21
        A = (155 + np.sqrt(15)) / 2400
        B = (155 - np.sqrt(15)) / 2400

        x = np.array([1 / 3, a, 1 - 2 * a, a, b, 1 - 2 * b, b])
        y = np.array([1 / 3, a, a, 1 - 2 * a, b, b, 1 - 2 * b])
        w = np.array([9 / 80, A, A, A, B, B, B])

        return x, y, w

    raise NotImplementedError("Only orders 1, ..., 6 are implemented")
