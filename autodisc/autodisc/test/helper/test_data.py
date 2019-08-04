import json
import numpy as np
from autodisc.helper.data import JSONNumpyEncoder, json_numpy_object_hook
from autodisc.helper.data import set_dict_default_values


def test_set_dict_default_values():

    # simple dict
    def_dict = {'a': 1, 'b': 2}
    trg_dict = {'b': 20, 'c': 30}
    test_dict = {'a': 1, 'b': 20, 'c': 30}

    new_dict = set_dict_default_values(trg_dict, def_dict)

    assert new_dict == test_dict


    # recursive dic
    def_dict = {'a': 1, 'b': {'aa': 5, 'bb': 6}}
    trg_dict = {'b': {'bb': 60, 'cc': 70}, 'c': 30}
    test_dict = {'a': 1, 'b': {'aa': 5, 'bb': 60, 'cc': 70}, 'c': 30}

    new_dict = set_dict_default_values(trg_dict, def_dict)

    assert new_dict == test_dict

    # empty dict
    def_dict = {'a': 1, 'b': {'aa': 5, 'bb': 6}}
    trg_dict = None

    new_dict = set_dict_default_values(trg_dict, def_dict)

    assert new_dict == def_dict



def test_json_numpy():
    expected = np.arange(100, dtype=np.float)
    dumped = json.dumps(expected, cls=JSONNumpyEncoder)
    result = json.loads(dumped, object_hook=json_numpy_object_hook)

    #None of the following assertions will be broken.
    assert result.dtype == expected.dtype, "Wrong Type"
    assert result.shape == expected.shape, "Wrong Shape"
    assert np.allclose(expected, result), "Wrong Values"

    expected = np.array([[1,2.3], [3, 4.3]])
    dumped = json.dumps(expected, cls=JSONNumpyEncoder)
    result = json.loads(dumped, object_hook=json_numpy_object_hook)

    #None of the following assertions will be broken.
    assert result.dtype == expected.dtype, "Wrong Type"
    assert result.shape == expected.shape, "Wrong Shape"
    assert np.allclose(expected, result), "Wrong Values"
