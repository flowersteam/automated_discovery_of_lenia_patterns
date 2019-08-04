import numpy as np
from pytest import approx
from autodisc.helper.statistics import calc_binary_segments
from autodisc.helper.statistics import calc_image_moments
from autodisc.helper.statistics import mean_over_angles_degrees
from autodisc.helper.statistics import nan_mean_over_angles_degrees
import autodisc as ad

def calc_central_moment(image, p, q):
    '''Explicit function to calculate the central moments of an image.'''

    size_x = image.shape[0]
    size_y = image.shape[1]

    y_grid, x_grid = np.meshgrid(range(size_y), range(size_x))

    m_00 = np.sum(image)
    m_10 = np.sum(image * x_grid)
    m_01 = np.sum(image * y_grid)

    x_avg = m_10 / m_00
    y_avg = m_01 / m_00

    return np.sum((x_grid - x_avg)**p * (y_grid - y_avg)**q * image)


def test_calc_image_moments():

    image = np.array([[1,2,3],
                      [4,5,6]])

    res = calc_image_moments(image)

    # first raw modes
    assert res.m00 == np.sum(image)
    assert res.m10 == np.sum(0*image[0,:] + 1*image[1,:])
    assert res.m01 == np.sum(0*image[:,0] + 1*image[:,1] + 2*image[:,2])

    # centroid
    assert res.y_avg == np.sum(0*image[0,:] + 1*image[1,:]) / np.sum(image)
    assert res.x_avg == np.sum(0 * image[:, 0] + 1 * image[:, 1] + 2 * image[:, 2]) / np.sum(image)

    # central moments
    mu_11_test = calc_central_moment(image, 1, 1)
    mu_20_test = calc_central_moment(image, 2, 0)
    mu_02_test = calc_central_moment(image, 0, 2)
    mu_30_test = calc_central_moment(image, 3, 0)
    mu_03_test = calc_central_moment(image, 0, 3)
    mu_21_test = calc_central_moment(image, 2, 1)
    mu_12_test = calc_central_moment(image, 1, 2)
    mu_22_test = calc_central_moment(image, 2, 2)
    mu_31_test = calc_central_moment(image, 3, 1)
    mu_13_test = calc_central_moment(image, 1, 3)
    mu_40_test = calc_central_moment(image, 4, 0)
    mu_04_test = calc_central_moment(image, 0, 4)

    assert res.mu11 == approx(mu_11_test)
    assert res.mu20 == approx(mu_20_test)
    assert res.mu02 == approx(mu_02_test)
    assert res.mu30 == approx(mu_30_test)
    assert res.mu03 == approx(mu_03_test)
    assert res.mu21 == approx(mu_21_test)
    assert res.mu12 == approx(mu_12_test)
    assert res.mu22 == approx(mu_22_test)
    assert res.mu31 == approx(mu_31_test)
    assert res.mu13 == approx(mu_13_test)
    assert res.mu40 == approx(mu_40_test)
    assert res.mu04 == approx(mu_04_test)

    # check hu invariant moments:
    # see https://en.wikipedia.org/wiki/Image_moment for definition
    hu_1 = res.eta20 + res.eta02
    hu_2 = (res.eta20 - res.eta02)**2 + 4 * res.eta11**2
    hu_3 = (res.eta30 - 3 * res.eta12)**2 + (3 * res.eta21 - res.eta03)**2
    hu_4 = (res.eta30 + res.eta12)**2 + (res.eta21 + res.eta03)**2
    hu_5 = (res.eta30 - 3*res.eta12)*(res.eta30 + res.eta12)*((res.eta30+res.eta12)**2 - 3*(res.eta21+res.eta03)**2) + (3*res.eta21-res.eta03)*(res.eta21+res.eta03)*(3*(res.eta30+res.eta12)**2 - (res.eta21 +res.eta03)**2)
    hu_6 = (res.eta20 - res.eta02)*((res.eta30 + res.eta12)**2 - (res.eta21 + res.eta03)**2) + 4*res.eta11*(res.eta30+res.eta12)*(res.eta21 + res.eta03)
    hu_7 = (3*res.eta21 - res.eta03)*(res.eta30+res.eta12)*((res.eta30+res.eta12)**2-3*(res.eta21 + res.eta03)**2) - (res.eta30 - 3*res.eta12)*(res.eta21+res.eta03)*(3*(res.eta30+res.eta12)**2 - (res.eta21+res.eta03)**2)

    assert res.hu1 == approx(hu_1)
    assert res.hu2 == approx(hu_2)
    assert res.hu3 == approx(hu_3)
    assert res.hu4 == approx(hu_4)
    assert res.hu5 == approx(hu_5)
    assert res.hu6 == approx(hu_6)
    assert res.hu7 == approx(hu_7)

    # check flusser and higher hu according to paper:
    # 2007 - Barczak, Johnson, Messom - Revisiting Moment Invariants: Rapid Feature Extraction and Classification for Handwritten Digits
    # Note that phi_6 = hu_8, phi_7 = flusser_9, phi_8 = flusser_10, phi_9 = flusser_11, phi_10 = flusser_12, phi_11 = flusser_13
    phi_6 = res.eta11 * ((res.eta30 + res.eta12)**2 - (res.eta03 + res.eta21)**2) - (res.eta20 - res.eta02) * (res.eta30 + res.eta12) * (res.eta03 + res.eta21)
    phi_7 = res.eta40 + res.eta04 + 2 * res.eta22
    phi_8 = (res.eta40 - res.eta04) * ((res.eta30 + res.eta12)**2 - (res.eta03 + res.eta21)**2) + 4 * (res.eta31 - res.eta13) * (res.eta30 - res.eta12) * (res.eta03 - res.eta21)
    phi_9 = 2 * (res.eta31 + res.eta13) * ((res.eta21 + res.eta03)**2 - (res.eta30 + res.eta12)**2) + 2 * (res.eta30 - res.eta12) * (res.eta21 - res.eta03) * (res.eta40 - res.eta04)
    phi_10 = (res.eta40 - 6 * res.eta22 + res.eta04) * ( ( (res.eta30 + res.eta12)**2 - (res.eta21 + res.eta03)**2 )**2 - 4 * (res.eta30 + res.eta12)**2 * (res.eta03 + res.eta21)**2 ) \
             + 16 * (res.eta31 - res.eta13) * (res.eta30 + res.eta12) * (res.eta03 + res.eta21) * ( (res.eta30 + res.eta12)**2 - (res.eta03 + res.eta21)**2 )
    phi_11 = 4 * (res.eta40 - 6 * res.eta22 + res.eta04) * (res.eta30 + res.eta12) * (res.eta03 + res.eta21) * ( (res.eta30 + res.eta12)**2 - (res.eta03 + res.eta21)**2) \
             - 4 * (res.eta31 - res.eta13) * ( ( (res.eta30 + res.eta12)**2 - (res.eta03 + res.eta21)**2 )**2 - 4 * (res.eta30 + res.eta12)**2 * (res.eta03 + res.eta21)**2)

    assert res.hu8 == approx(phi_6)
    assert res.flusser9 == approx(phi_7)
    assert res.flusser10 == approx(phi_8)
    assert res.flusser11 == approx(phi_9)
    assert res.flusser12 == approx(phi_10)
    assert res.flusser13 == approx(phi_11)

    # rotation of the image should not change the computed moments
    image2 = np.array([[4, 1],
                      [5, 2],
                      [6, 3]])

    res2 = calc_image_moments(image2)

    assert res.hu1 == approx(res2.hu1)
    assert res.hu2 == approx(res2.hu2)
    assert res.hu3 == approx(res2.hu3)
    assert res.hu4 == approx(res2.hu4)
    assert res.hu5 == approx(res2.hu5)
    assert res.hu6 == approx(res2.hu6)
    assert res.hu7 == approx(res2.hu7)
    assert res.hu8 == approx(res2.hu8)

    assert res.flusser9 == approx(res2.flusser9, abs=1e-8)
    assert res.flusser10 == approx(res2.flusser10, abs=1e-8)
    assert res.flusser11 == approx(res2.flusser11, abs=1e-8)
    assert res.flusser12 == approx(res2.flusser12, abs=1e-8)
    assert res.flusser13 == approx(res2.flusser13, abs=1e-8)


    #####################################################################################################
    # check that the center position is the mid of the image if all values are equal

    image = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])

    res = calc_image_moments(image)

    # first raw modes
    assert res.m00 == np.sum(image)
    assert res.y_avg == 1
    assert res.x_avg == 1


    # check that the center position is the mid of the image if all values are equal

    image = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])

    res = calc_image_moments(image)

    # first raw modes
    assert res.m00 == np.sum(image)
    assert res.y_avg == 1
    assert res.x_avg == 1


def test_mean_over_angle():

    angles = [0, 30, 60]
    angle_mean = mean_over_angles_degrees(angles)
    assert angle_mean == approx(30)

    angles = [35, 5, 335]
    angle_mean = mean_over_angles_degrees(angles)
    assert angle_mean == approx(5)

    angles = [90, 100, 110]
    angle_mean = mean_over_angles_degrees(angles)
    assert angle_mean == approx(100)


def test_nan_mean_over_angle():

    angles = [np.nan, 0, 30, 60]
    angle_mean = nan_mean_over_angles_degrees(angles)
    assert angle_mean == approx(30)

    angles = np.array([35, np.nan, 5, 335])
    angle_mean = nan_mean_over_angles_degrees(angles)
    assert angle_mean == approx(5)

    angles = [90, np.nan, 100, 110, np.nan]
    angle_mean = nan_mean_over_angles_degrees(angles)
    assert angle_mean == approx(100)


def test_calc_binary_segments():

    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img)
    assert len(num_of_segments) == 1
    assert num_of_segments[0] == 8*10
    assert seg_types[0] == False
    assert np.all(result_seg == exp)

    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img)
    assert len(num_of_segments) == 2
    assert num_of_segments[0] == 8*10-4
    assert num_of_segments[1] == 4
    assert seg_types[0] == False
    assert seg_types[1] == True
    assert np.all(result_seg==exp)


    ################################################################
    img = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img)

    assert len(num_of_segments) == 3
    assert num_of_segments[0] == 4
    assert num_of_segments[1] == 8*10-8
    assert num_of_segments[2] == 4
    assert seg_types[0] == True
    assert seg_types[1] == False
    assert seg_types[2] == True
    assert np.all(result_seg==exp)


    ################################################################
    img = np.array([[1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0,0.5, 0]])

    exp = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 1]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img)

    assert len(num_of_segments) == 2
    assert num_of_segments[0] == 9
    assert num_of_segments[1] == 16
    assert seg_types[0] == True
    assert seg_types[1] == False
    assert np.all(result_seg==exp)


    ################################################################
    img = np.array([[0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0]])

    exp = np.array([[0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 3, 0],
                    [0, 1, 0, 0, 0]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img)

    assert len(num_of_segments) == 4
    assert num_of_segments[0] == 16
    assert num_of_segments[1] == 7
    assert num_of_segments[2] == 1
    assert num_of_segments[3] == 1
    assert seg_types[0] == False
    assert seg_types[1] == True
    assert seg_types[2] == True
    assert seg_types[3] == True
    assert np.all(result_seg==exp)


    ################################################################
    img = np.array([[0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0]])

    exp = np.array([[0, 1, 2, 3, 4],
                    [1, 1, 2, 3, 3],
                    [5, 5, 6, 5, 5],
                    [5, 5, 5, 7, 5],
                    [5, 8, 5, 5, 5]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img, is_continuous_image=False)

    assert len(num_of_segments) == 9
    assert num_of_segments[0] == 1
    assert num_of_segments[1] == 3
    assert num_of_segments[2] == 2
    assert num_of_segments[3] == 3
    assert num_of_segments[4] == 1
    assert num_of_segments[5] == 12
    assert num_of_segments[6] == 1
    assert num_of_segments[7] == 1
    assert num_of_segments[8] == 1
    assert seg_types[0] == False
    assert seg_types[1] == True
    assert seg_types[2] == False
    assert seg_types[3] == True
    assert seg_types[4] == False
    assert seg_types[5] == False
    assert seg_types[6] == True
    assert seg_types[7] == True
    assert seg_types[8] == True

    assert np.all(result_seg==exp)

    ################################################################
    img = np.array([[0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0]])

    exp = np.array([[0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img, is_eight_neighbors=True)

    assert len(num_of_segments) == 3
    assert num_of_segments[0] == 16
    assert num_of_segments[1] == 8
    assert num_of_segments[2] == 1
    assert seg_types[0] == False
    assert seg_types[1] == True
    assert seg_types[2] == True
    assert np.all(result_seg==exp)


    ################################################################
    img = np.array([[0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0]])

    exp = np.array([[0, 1, 2, 1, 3],
                    [1, 1, 2, 1, 1],
                    [2, 2, 1, 2, 2],
                    [2, 2, 2, 2, 4],
                    [2, 5, 2, 2, 2]])

    [result_seg, seg_types, num_of_segments] = calc_binary_segments(img, is_eight_neighbors=True, is_continuous_image=False)

    assert len(num_of_segments) == 6
    assert num_of_segments[0] == 1
    assert num_of_segments[1] == 7
    assert num_of_segments[2] == 14
    assert num_of_segments[3] == 1
    assert num_of_segments[4] == 1
    assert num_of_segments[5] == 1
    assert seg_types[0] == False
    assert seg_types[1] == True
    assert seg_types[2] == False
    assert seg_types[3] == False
    assert seg_types[4] == True
    assert seg_types[5] == True
    assert np.all(result_seg==exp)


    ################################################################
    img = np.array([[0, 0, 0.3, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0.4, 0.3, 0, 0]])

    exp = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0]])

    [result_seg, _, _] = calc_binary_segments(img, is_eight_neighbors=False, is_continuous_image=False)

    assert np.all(result_seg==exp)

    ################################################################
    img = np.array([[0,   0, 0.4],
                    [0.3, 0, 0.3]])

    exp = np.array([[0, 0, 1],
                    [2, 0, 1]])

    [result_seg, _, _] = calc_binary_segments(img, is_eight_neighbors=False, is_continuous_image=False)

    assert np.all(result_seg == exp)

    ################################################################
    img = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.4],
                    [0.3, 0, 0, 0, 0.3],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [2, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])

    [result_seg, _, _] = calc_binary_segments(img, is_eight_neighbors=False, is_continuous_image=False)

    assert np.all(result_seg==exp)


def test_calc_space_distribution_histogram():

    vectors = [[0.5, 0.5],
               [1.5, 0.5],
               [0.5, 1.5],
               [0.5, 1.5]]

    bins = [(0, 2, 2),
            (0, 2, 2)]

    hist, _ = ad.helper.statistics.calc_space_distribution_histogram(vectors, bins)
    assert np.all(hist == [1,2,1])


    bins = [(0, 2, 2),
            (0, 2, 1)]

    hist, _ = ad.helper.statistics.calc_space_distribution_histogram(vectors, bins)
    assert np.all(hist == [0,1,0,1])

    ######################################################
    # ignore counting of outside points
    vectors = [[0.5, 0.5],
               [1.5, 0.5],
               [0.5, 1.5],
               [0.5, 1.5],
               [3.0, 1],
               [0.5, -1]]

    bins = [(0, 2, 2),
            (0, 2, 2)]

    hist, _ = ad.helper.statistics.calc_space_distribution_histogram(vectors, bins)
    assert np.all(hist == [1,2,1])

    ######################################################
    # count outside points

    vectors = [[0.5, 0.5],
               [1.5, 0.5],
               [0.5, 1.5],
               [0.5, 1.5],
               [3.0, 1],
               [0.5, -1]]

    bins = [(0, 2, 2),
            (0, 2, 2)]

    hist, _ = ad.helper.statistics.calc_space_distribution_histogram(vectors, bins, ignore_out_of_range_values=False)
    assert np.all(hist == [11,4,1])


def test_calc_space_distribution_bins():

    vectors = [[0.5, 0.5],
               [1.5, 0.5],
               [0.5, 1.5],
               [0.5, 1.5]]

    bins = [(0, 2, 2),
            (0, 2, 2)]

    bins_descr = ad.helper.statistics.calc_space_distribution_bins(vectors, bins)

    assert np.all(bins_descr['n_points'] == [2,1,1])
    assert np.all(bins_descr['upper_borders'] == [[1, 2], [2, 1], [1, 1]])
    assert np.all(bins_descr['lower_borders'] == [[0, 1], [1, 0], [0, 0]])

    bins = [(0, 2, 2),
            (0, 2, 1)]


    bins_descr = ad.helper.statistics.calc_space_distribution_bins(vectors, bins)

    assert np.all(bins_descr['n_points'] == [3,1])
    assert np.all(bins_descr['upper_borders'] == [[1, 2], [2, 2]])
    assert np.all(bins_descr['lower_borders'] == [[0, 0], [1, 0]])


    ######################################################
    # ignore counting of outside points
    vectors = [[0.5, 0.5],
               [1.5, 0.5],
               [0.5, 1.5],
               [0.5, 1.5],
               [3.0, 1],
               [0.5, -1]]

    bins = [(0, 2, 2),
            (0, 2, 2)]

    bins_descr = ad.helper.statistics.calc_space_distribution_bins(vectors, bins)

    assert np.all(bins_descr['n_points'] == [2,1,1])
    assert np.all(bins_descr['upper_borders'] == [[1, 2], [2, 1], [1, 1]])
    assert np.all(bins_descr['lower_borders'] == [[0, 1], [1, 0], [0, 0]])


    ######################################################
    # count outside points

    vectors = [[0.5, 0.5],
               [1.5, 0.5],
               [0.5, 1.5],
               [0.5, 1.5],
               [3.0, 1.5],
               [0.5, -1]]

    bins = [(0, 2, 2),
            (0, 2, 2)]


    bins_descr = ad.helper.statistics.calc_space_distribution_bins(vectors, bins, ignore_out_of_range_values=False)

    assert np.all(bins_descr['n_points'] == [2, 1, 1, 1, 1])
    assert np.all(bins_descr['upper_borders'] == [[1, 2], [1, 0],      [np.nan, 2], [2, 1], [1, 1]])
    assert np.all(bins_descr['lower_borders'] == [[0, 1], [0, np.nan], [2, 1],      [1, 0], [0, 0]])


# def test_is_finite_object():
#
#     #########################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == True
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == True
#
#     #########################################################
#     img = np.array([[0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == True
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == False
#
#     #########################################################
#     img = np.array([[0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == False
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == False
#
#
#     #########################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [0, 1, 1, 0, 0],
#                     [0, 1, 1, 0, 0],
#                     [0, 1, 1, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == True
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == True
#
#
#     #########################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 1, 1, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == True
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == True
#
#     #########################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0],
#                     [1, 1, 1, 1, 1],
#                     [0, 1, 1, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == False
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == False
#
#     #########################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0],
#                     [1, 1, 1, 1, 0],
#                     [0, 1, 1, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == True
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == False
#
#     #########################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [1, 1, 0, 1, 1],
#                     [0, 1, 1, 1, 0],
#                     [0, 1, 1, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == False
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == False
#
#     #########################################################
#     img = np.array([[0, 1, 0, 1, 0],
#                     [1, 1, 0, 1, 1],
#                     [0, 0, 0, 0, 0],
#                     [1, 1, 0, 1, 1],
#                     [0, 1, 0, 1, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == True
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == False
#
#
#     #########################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [1, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0]])
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=True)
#     assert is_finite == True
#
#     is_finite = ad.helper.statistics.is_finite_object(img, is_continuous_image=False)
#     assert is_finite == False


def test_calc_active_binary_segments():

    ################################################################
    ################################################################
    # CONTINUOUS

    ################################################################
    img = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=2)

    assert np.all(result_seg == exp)

    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 2, 0],
                    [0, 1, 0, 1, 0, 2, 0],
                    [0, 1, 1, 1, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=1)

    assert np.all(result_seg == exp)


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 0, 1, 1],
                    [0, 1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [1, 1, 0, 1, 0, 1, 1],
                    [0, 1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=1)

    assert np.all(result_seg == exp)


    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=2)

    assert np.all(result_seg == exp)


    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 0, 2],
                    [0, 3, 3, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=1)

    assert np.all(result_seg == exp)


    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0]])

    exp = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=2)

    assert np.all(result_seg == exp)

    ################################################################
    ################################################################
    # NON CONTINUOUS

    ################################################################
    img = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=2, is_continuous_image=False)

    assert np.all(result_seg == exp)

    ################################################################
    img = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0]])

    exp = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 2],
                    [0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=2, is_continuous_image=False)

    assert np.all(result_seg == exp)

    ################################################################
    img = np.array([[0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0]])

    exp = np.array([[0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 2],
                    [0, 0, 0, 0, 0]])

    result_seg, _ = ad.helper.statistics.calc_active_binary_segments(img, r=2, is_continuous_image=False)

    assert np.all(result_seg == exp)


# def test_BinareImageSegmenter():
#
#     ################################################################
#     img = np.array([[0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0],
#                     [0, 0, 0, 0, 0]])
#
#     exp = np.array([[0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0],
#                     [0, 0, 0, 0, 0]])
#
#     segmenter = ad.helper.statistics.BinaryImageSegmenter(r=2)
#
#     result_seg = segmenter.calc(img)
#
#     assert np.all(result_seg == exp)
#
#     ################################################################
#     img = np.array([[0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0, 1, 0],
#                     [0, 1, 0, 1, 0, 1, 0],
#                     [0, 1, 1, 1, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])
#
#     exp = np.array([[0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0, 2, 0],
#                     [0, 1, 0, 1, 0, 2, 0],
#                     [0, 1, 1, 1, 0, 2, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])
#
#     segmenter = ad.helper.statistics.BinaryImageSegmenter(r=1)
#
#     result_seg = segmenter.calc(img)
#
#     assert np.all(result_seg == exp)
#
#
#     ################################################################
#     img = np.array([[0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0, 1, 0],
#                     [1, 1, 0, 1, 0, 1, 1],
#                     [0, 1, 1, 1, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])
#
#     exp = np.array([[0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 1, 0, 1, 0],
#                     [1, 1, 0, 1, 0, 1, 1],
#                     [0, 1, 1, 1, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])
#
#     segmenter = ad.helper.statistics.BinaryImageSegmenter(r=1)
#
#     result_seg = segmenter.calc(img)
#
#     assert np.all(result_seg == exp)
#
#
#     ################################################################
#     img = np.array([[0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [1, 0, 0, 0, 0, 1, 0],
#                     [0, 1, 1, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])
#
#     exp = np.array([[0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [1, 0, 0, 0, 0, 1, 0],
#                     [0, 1, 1, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])
#
#     segmenter = ad.helper.statistics.BinaryImageSegmenter(r=2)
#
#     result_seg = segmenter.calc(img)
#
#     assert np.all(result_seg == exp)
#
#
#     ################################################################
#     img = np.array([[0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 1],
#                     [0, 1, 1, 0, 0, 1],
#                     [0, 0, 0, 0, 0, 0]])
#
#     exp = np.array([[0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 2],
#                     [2, 0, 0, 0, 0, 2],
#                     [0, 3, 3, 0, 0, 2],
#                     [0, 0, 0, 0, 0, 0]])
#
#     segmenter = ad.helper.statistics.BinaryImageSegmenter(r=1)
#
#     result_seg = segmenter.calc(img)
#
#     assert np.all(result_seg == exp)
#
#
#     ################################################################
#     img = np.array([[0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 1],
#                     [0, 1, 1, 0, 0, 1],
#                     [0, 0, 0, 0, 0, 0]])
#
#     exp = np.array([[0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1],
#                     [1, 0, 0, 0, 0, 1],
#                     [0, 1, 1, 0, 0, 1],
#                     [0, 0, 0, 0, 0, 0]])
#
#     segmenter = ad.helper.statistics.BinaryImageSegmenter(r=2)
#
#     result_seg = segmenter.calc(img)
#
#     assert np.all(result_seg == exp)


def test_calc_is_segment_finite():

    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    r = 1

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == [1]


    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == [1]


    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == [1,2]


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _, _ = ad.helper.statistics.calc_is_segments_finite(img, r=r)

    assert finite_segments == []


def test_calc_is_segment_finite_v2():

    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    r = 1

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == [1]


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == [1]


    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == [1,2]


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
    r = 2

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == []


    ################################################################
    img = np.array([[0, 1, 0, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0]])

    r = 1

    # 1. identify all possible animals
    finite_segments, _ = ad.helper.statistics.calc_is_segments_finite_v2(img, r=r)

    assert finite_segments == [1]