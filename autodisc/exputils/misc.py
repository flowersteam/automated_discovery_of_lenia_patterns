import numpy as np

def numpy_vstack_2d_default(array1, array2, default_value=np.nan):

    if len(array1) == 0:
        return array2

    if len(array2) == 0:
        return array1

    if np.ndim(array1) == 1:
        array1 = np.reshape(array1, (1,len(array1)))

    if np.ndim(array2) == 1:
        array2 = np.reshape(array2, (1,len(array2)))

    shape1 = np.shape(array1)
    shape2 = np.shape(array2)

    if shape1[1] == shape2[1]:
        return np.vstack((array1, array2))

    elif shape1[1] > shape2[1]:
        # add default values to array1

        new_values = np.ones((shape2[0], shape1[1] - shape2[1]))
        new_values[:] = default_value

        return np.vstack((array1, np.hstack((array2, new_values))))

    else:
        # add default values to array2

        new_values = np.ones((shape1[0], shape2[1] - shape1[1]))
        new_values[:] = default_value

        return np.vstack((np.hstack((array1, new_values)), array2))
