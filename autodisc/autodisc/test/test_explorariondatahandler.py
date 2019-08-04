import autodisc as ad
import numpy as np

def test_database(tmpdir):


    ###########################################################################
    # test error for direct usage of handler
    try:
        dbh = ad.ExplorationDataHandler()
        raise Exception('This should have produce an exception!')
    except NotImplementedError:
        pass

    ###########################################################################

    # make database
    dbh = ad.ExplorationDataHandler.create(db_type='memory')

    # add experiment data
    data = {'sys_params': [1, 2], 'name': 'experiment'}
    dbh.add_exploration_data(data)

    assert dbh.data['sys_params'] == [1, 2]
    assert dbh.data.sys_params == [1, 2]
    assert dbh.sys_params == [1, 2]
    assert dbh['sys_params'] == [1, 2]
    assert dbh.data['name'] == 'experiment'
    assert dbh.data.name == 'experiment'

    dbh.add_exploration_data(sys_params=[3, 4])

    assert dbh.data['sys_params'] == [3, 4]
    assert dbh.data.sys_params == [3, 4]
    assert dbh.data['name'] == 'experiment'
    assert dbh.data.name == 'experiment'

    data = {'name': 'experiment2'}
    dbh.add_exploration_data(data, sys_params=[5, 6])

    assert dbh.data['sys_params'] == [5, 6]
    assert dbh.data.sys_params == [5, 6]
    assert dbh.data['name'] == 'experiment2'
    assert dbh.data.name == 'experiment2'


    ###########################################################################
    # add exploration data

    data = {'name': 'exploration', 'seed': 1}
    dbh.add_run_data(0, data)

    assert dbh[0]['name'] == 'exploration'
    assert dbh[0].name == 'exploration'
    assert dbh[0]['seed'] == 1
    assert dbh[0].seed == 1

    dbh.add_run_data(0, seed=2)

    assert dbh[0]['name'] == 'exploration'
    assert dbh[0].name == 'exploration'
    assert dbh[0]['seed'] == 2
    assert dbh[0].seed == 2

    data = {'name': 'exploration2'}
    dbh.add_run_data(1, data, seed=3)

    assert dbh[1]['name'] == 'exploration2'
    assert dbh[1].name == 'exploration2'
    assert dbh[1]['seed'] == 3
    assert dbh[1].seed == 3

    del dbh


    #############################################################################
    # test save

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath)

    assert dbh.config.directory == tmpdir.strpath

    # add all data fields
    ed = ad.DataEntry()
    ed.statistics = {'stat1': np.array([1,2,3,4]), 'stat2': np.array([10,20,30,40])}
    ed.id = 0
    ed.seed = 1
    ed.name = 'experiment1'
    ed.descr = 'Description of ed'
    ed.system_parameters = np.array([1,2,3])

    dbh.add_exploration_data(ed)


    # add 2 runs with all data
    exp1 = ad.DataEntry()
    exp1.seed = 1
    exp1.run_parameters = np.array([1,2,3,4])
    exp1.observations = {'states':[np.array([1,2,3,4,5]), np.array([10,20,30,40])], 'timepoints': np.array([0, 1])}
    exp1.statistics = {'stat1': np.array([-1,-2,-3]), 'stat2': np.array([-10,-20,-30])}
    exp1.goal_space = np.array([10, 20])

    dbh.add_run_data(1, exp1)

    dbh.save()

    exp2 = ad.DataEntry()
    exp2.seed = 2
    exp2.run_parameters = np.array([10,20,30,40])
    exp2.observations = {'states':[np.array([10,20,30,40,50]), np.array([100,200,300,400])], 'timepoints': np.array([0, 1])}
    exp2.statistics = {'stat1': np.array([-10,-20,-30]), 'stat2': np.array([-100,-200,-300])}
    exp2.goal_space = np.array([100, 200])

    dbh.add_run_data(2, exp2)

    dbh.save()


    data_before = dbh.data.copy()

    del dbh


    #############################################################################
    # load
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath)
    dbh.load()

    assert dd(dbh.data, data_before)

    del dbh


    #############################################################################
    # load without observations
    data_without_obs = data_before.copy()

    data_without_obs.runs[1]['observations'] = None
    data_without_obs.runs[2]['observations'] = None

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, load_observations=False)
    dbh.load()

    assert dd(dbh.data, data_without_obs)

    del dbh


    #############################################################################
    # load with restricted memory

    data_without_all_rundata = data_before.copy()

    del data_without_all_rundata.runs[2]

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, memory_size_run_data=1)
    dbh.load()

    assert dd(dbh.data, data_without_all_rundata)

    del dbh



    #############################################################################
    # automatically save
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True)
    dbh.add_exploration_data(ed)
    dbh.add_run_data(1, exp1)
    dbh.add_run_data(2, exp2)

    del dbh

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath)
    dbh.load()

    assert dd(dbh.data, data_before)

    del dbh


    #############################################################################
    # don't keep memory

    data_without_runs = data_before.copy()
    data_without_runs.runs.clear()

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True, keep_saved_runs_in_memory=False)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    assert dd(dbh.data, data_without_runs)

    dbh.add_run_data(2, exp2)
    assert dd(dbh.data, data_without_runs)

    del dbh

    # after loading has the data
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath)
    dbh.load()

    assert dd(dbh.data, data_before)

    del dbh


    #############################################################################
    # don't keep observations

    data_without_runs = data_before.copy()
    data_without_runs.runs.clear()

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True, keep_saved_observations_in_memory=False, load_observations=False)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    dbh.add_run_data(2, exp2)

    assert dd(dbh.data, data_without_obs)

    del dbh

    # after loading has the data
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath)
    dbh.load()

    assert dd(dbh.data, data_before)

    del dbh

    #############################################################################
    # reload observations if they are requested

    data_without_runs = data_before.copy()
    data_without_runs.runs.clear()

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True, keep_saved_observations_in_memory=False, load_observations=True)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    dbh.add_run_data(2, exp2)

    assert dd(dbh.data, data_before)

    del dbh


    #############################################################################
    # reload run_data if they are requested

    data_without_runs = data_before.copy()
    data_without_runs.runs.clear()

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True, keep_saved_runs_in_memory=False)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    dbh.add_run_data(2, exp2)

    assert len(dbh) == 2
    assert len(dbh.data.runs) == 0
    assert dd(dbh[1], data_before.runs[1])
    assert dd(dbh[2], data_before.runs[2])

    del dbh


    ############################################################################
    # restricted memory for run data
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True, memory_size_run_data=1)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    dbh.add_run_data(2, exp2)

    assert len(dbh) == 2
    assert len(dbh.data.runs) == 1
    assert dd(dbh[1], data_before.runs[1])
    assert dd(dbh[2], data_before.runs[2])
    assert dd(dbh[1], data_before.runs[1])
    assert len(dbh.data.runs) == 1

    del dbh


    ############################################################################
    # restricted memory for observations
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True, memory_size_observations=1)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    dbh.add_run_data(2, exp2)

    assert len(dbh) == 2
    assert len(dbh.data.runs) == 2
    assert dd(dbh[1], data_before.runs[1])
    assert dd(dbh[2], data_before.runs[2])
    assert dd(dbh[1], data_before.runs[1])
    assert len(dbh.data.runs) == 2

    del dbh

    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=True, load_observations=False, memory_size_observations=1)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    dbh.add_run_data(2, exp2)

    assert dbh[1].observations is None
    assert dd(dbh[2].observations, data_before.runs[2].observations)

    del dbh

    ############################################################################
    # combination automatic saving and memory size - run data
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=2, memory_size_run_data=1)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    assert len(dbh) == 1
    assert len(dbh.data.runs) == 1
    assert dbh.unsaved_run_ids == {1}

    dbh.add_run_data(2, exp2)
    assert len(dbh) == 2
    assert len(dbh.data.runs) == 1
    assert dbh.unsaved_run_ids == {2}    # only save first, because of smaller memory size

    # if first in loaded, back, then then second must be saved, because of reduced memory size
    assert dd(dbh[1], data_before.runs[1])
    assert len(dbh) == 2
    assert len(dbh.data.runs) == 1
    assert dbh.unsaved_run_ids == set()  # only save first, because of smaller memory size

    assert dd(dbh[2], data_before.runs[2])

    del dbh

    ############################################################################
    # combination automatic saving and memory size - observations
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath, save_automatic=2, memory_size_observations=1)
    dbh.add_exploration_data(ed)

    dbh.add_run_data(1, exp1)
    assert len(dbh) == 1
    assert len(dbh.data.runs) == 1
    assert dbh.unsaved_run_ids == {1}

    dbh.add_run_data(2, exp2)
    assert len(dbh) == 2
    assert len(dbh.data.runs) == 2      # keep run data of all runs
    assert dbh.unsaved_run_ids == {2}    # only save first, because of smaller memory size

    # if first in loaded, back, then then second must be saved, because of reduced memory size
    assert dd(dbh[1], data_before.runs[1])
    assert len(dbh) == 2
    assert len(dbh.data.runs) == 2      # keep run data of all runs
    assert dbh.unsaved_run_ids == set()  # only save first, because of smaller memory size

    assert dd(dbh[2], data_before.runs[2])

    del dbh

    ##############################################################
    # test special data operations
    dbh = ad.ExplorationDataHandler.create(directory=tmpdir.strpath)
    dbh.load()

    assert 'descr' in dbh
    assert 'bla' not in dbh

    assert 1 in dbh
    assert 2 in dbh
    assert 3 not in dbh

    i = 1
    for run in dbh:
        assert run.id == i
        i += 1

    assert len(dbh) == 2


def test_filter():
    dbh = ad.ExplorationDataHandler.create(db_type='memory')

    # add all data fields

    # add 2 runs with all data
    exp1 = ad.DataEntry()
    exp1.seed = 1
    exp1.run_parameters = np.array([1, 2, 3, 4])
    exp1.observations = {'states': [np.array([10, 20, 30, 40, 50]), np.array([100, 200, 300, 400])], 'timepoints': np.array([0, 1])}
    exp1.statistics = {'stat1': True, 'stat2': False}
    exp1.goal_space = np.array([10, 20])

    dbh.add_run_data(1, exp1)

    exp2 = ad.DataEntry()
    exp2.seed = 2
    exp2.run_parameters = np.array([10, 20, 30, 40])
    exp2.observations = {'states': [np.array([10, 20, 30, 40, 50]), np.array([100, 200, 300, 400])], 'timepoints': np.array([0, 1])}
    exp2.statistics = {'stat1': False, 'stat2': False}
    exp2.goal_space = np.array([100, 200])

    dbh.add_run_data(2, exp2)

    filterd_inds = dbh.filter(('statistics.stat1', '==', True))
    assert np.all(filterd_inds == np.array([True, False]))

    filterd_inds = dbh.filter((('statistics.stat1', '==', True), 'and', ('statistics.stat2', '==', False)))
    assert np.all(filterd_inds == np.array([True, False]))

    filterd_inds = dbh.filter((('statistics.stat1', '==', True), 'and', ('statistics.stat2', '==', True)))
    assert np.all(filterd_inds == np.array([False, False]))



def dd(d1, d2, ctx=""):
    '''Tests if two dictionaries have the same data'''

    same = True

    def list_to_dict(l):
        return dict(zip(map(str, range(len(l))), l))

    for k in d1:
        if k not in d2:
            print(k + " removed from d2")
            return False
    for k in d2:
        if k not in d1:
            print(str(k) + " added in d2")
            return False

        if isinstance(d2[k], dict):
            same &= dd(d1[k], d2[k], ctx)
        elif isinstance(d2[k], list):
            same &= dd(list_to_dict(d1[k]), list_to_dict(d2[k]), ctx)
        else:
            if isinstance(d2[k], np.ndarray):
                d2[k] = d2[k].tolist()
                d1[k] = d1[k].tolist()

            # ignore datahandler entry
            if k != 'datahandler' and d2[k] != d1[k]:
                if type(d2[k]) not in (dict, list):
                    print( k + " changed in d2 to " + str(d2[k]))
                    return False

                else:
                    if type(d1[k]) != type(d2[k]):
                        print(k+ " changed to " + str(d2[k]))
                        return False

    return same