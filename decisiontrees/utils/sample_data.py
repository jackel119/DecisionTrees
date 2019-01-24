import numpy as np


def gen_quadrants_data(no_points):

    def quadrant(row):
        if row[0] > 0:
            if row[1] > 0:
                return 1
            else:
                return 4
        else:
            if row[1] > 0:
                return 2
            else:
                return 3

    arr = np.random.rand(no_points, 2) * 200
    arr = arr - 100

    labels = np.array(list(map(quadrant, arr))).reshape((no_points, 1))

    labeled_data = np.append(arr, labels, axis=1)

    print(labeled_data)

    return labeled_data
