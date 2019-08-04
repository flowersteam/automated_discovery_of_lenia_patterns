import autodisc as ad
import numpy as np

def test_stop_conditions():

    # use the lenia system
    lenia = ad.systems.Lenia(statistics=[])


    # max number of steps
    stop_conditions = 3

    [observations, _] = lenia.run(stop_conditions=stop_conditions)

    assert len(observations.states) == 3


    # max number of steps as list
    stop_conditions = [3,2]

    [observations, _] = lenia.run(stop_conditions=stop_conditions)

    assert len(observations.states) == 2


    # functions
    func = lambda system, step, observation, statistics: step == 4

    stop_conditions = [10, func]

    [observations, _] = lenia.run(stop_conditions=stop_conditions)

    assert len(observations.states) == 5



def test_representations_distance_function():

    rep = ad.core.Representation()

    dist = rep.calc_distance([1, 1], [2, 2])
    assert dist == np.linalg.norm([1,1])

    dist = rep.calc_distance([[1, 1],[3, 4]], [[2, 2],[5, 3]])
    assert np.all(dist == np.linalg.norm([[1, 1],[2, 1]], axis=1))

    dist = rep.calc_distance([], [])
    assert dist.size == 0

    dist = rep.calc_distance([1, 1], [[2, 2], [5, 3]])
    assert np.all(dist == np.linalg.norm([[1, 1], [4, 2]], axis=1))