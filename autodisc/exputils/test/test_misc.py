import exputils
import numpy as np

def test_numpy_vstack_2d_default():

    mat1 = [1, 2, 3, 4]
    mat2 = [10, 20, 30, 40]

    result = exputils.misc.numpy_vstack_2d_default(mat1, mat2)

    trg = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])

    assert np.all(result == trg)


    #################

    mat1 = [1, 2, 3]
    mat2 = [10, 20, 30, 40]

    result = exputils.misc.numpy_vstack_2d_default(mat1, mat2)

    trg = np.array([[1, 2, 3, np.nan], [10, 20, 30, 40]])

    np.testing.assert_equal(result, trg)


    #################

    mat1 = [1, 2, 3, 4]
    mat2 = [10, 20, 30]

    result = exputils.misc.numpy_vstack_2d_default(mat1, mat2)

    trg = np.array([[1, 2, 3, 4], [10, 20, 30, np.nan]])

    np.testing.assert_equal(result, trg)


    #################

    mat1 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    mat2 = [10, 20, 30]

    result = exputils.misc.numpy_vstack_2d_default(mat1, mat2)

    trg = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [10, 20, 30, np.nan]])

    np.testing.assert_equal(result, trg)


    #################

    mat1 = []
    mat2 = [10, 20, 30]

    result = exputils.misc.numpy_vstack_2d_default(mat1, mat2)

    trg = np.array([10, 20, 30])

    np.testing.assert_equal(result, trg)