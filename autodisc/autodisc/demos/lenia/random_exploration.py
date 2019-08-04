#!/usr/bin/env python

import os
import autodisc as ad


def run_exploration():
    # define exploration parameters

    directory = os.path.join(os.path.dirname(__file__), "results_random_exploration")

    system_parameters = ad.systems.Lenia.default_system_parameters()
    system_parameters.size_y = 100
    system_parameters.size_x = 100

    lenia = ad.systems.Lenia(system_parameters=system_parameters)

    config = ad.explorers.RandomExplorer.default_config()
    config.num_of_steps = 100

    random_init_state_config = dict()
    random_init_state_config['size_x'] = lenia.system_parameters['size_x']
    random_init_state_config['size_y'] = lenia.system_parameters['size_y']
    random_init_state_config['min_value'] = 0
    random_init_state_config['max_value'] = 1
    random_init_state_config['num_of_blobs'] = ('discrete', 1, 5)
    random_init_state_config['is_uniform_blobs'] = False
    random_init_state_config['blob_radius'] = (5, 100)
    random_init_state_config['blob_position_x'] = None  # if none, then random
    random_init_state_config['blob_position_y'] = None  # if none, then random
    random_init_state_config['blob_height'] = (0.3, 1)
    random_init_state_config['sigma'] = (0.7, 10)  # if none --> no gradient
    random_init_state_config['is_image_repeating'] = True

    config.explore_parameters['init_state'] = ('function', ad.helper.sampling.sample_bubble_image, random_init_state_config)  # function, config
    config.explore_parameters['b'] = ('function', ad.helper.sampling.sample_vector, (('discrete', 1, 3), (0, 1)))  # function, config
    config.explore_parameters['R'] = {'type': 'discrete', 'min': 2, 'max': 50}
    config.explore_parameters['T'] = {'type': 'continuous', 'min': 1, 'max': 50}
    config.explore_parameters['m'] = {'type': 'continuous', 'min': 0, 'max': 1}
    config.explore_parameters['s'] = {'type': 'continuous', 'min': 0, 'max': 1}
    config.explore_parameters['kn'] = {'type': 'discrete', 'min': 0, 'max': 3}
    config.explore_parameters['gn'] = {'type': 'discrete', 'min': 0, 'max': 2}

    explorer = ad.explorers.RandomExplorer(system=lenia, config=config)
    explorer.data.config.save_automatic = True
    explorer.data.config.keep_saved_runs_in_memory = True
    explorer.data.config.keep_saved_observations_in_memory = False
    explorer.data.config.directory = directory

    # run exploration
    explorer.run(num_of_runs=5, verbose=True)
    explorer.save()

    return explorer


def view_exploration_results(explorer):

    gui_config = ad.gui.ExplorationGUI.default_gui_config()

    gui_config['max_num_of_obs_in_memory'] = 20
    gui_config['experiment_num_of_steps'] = 300

    gui_config['statistic_columns'] = []
    gui_config['statistic_columns'].append({'stat_name': 'activation_mass_mean', 'disp_name': 'act mass', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'positive_growth_mass_mean', 'disp_name': 'growth mass', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_volume_mean', 'disp_name': 'act volume', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'positive_growth_volume_mean', 'disp_name': 'growth volume', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_density_mean', 'disp_name': 'act density', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'positive_growth_density_mean', 'disp_name': 'growth density', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_center_velocity_mean', 'disp_name': 'act centroid speed', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_positive_growth_centroid_distance_mean', 'disp_name': 'act-growth centroid distance', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_center_movement_angle_velocity_mean', 'disp_name': 'act centroid rotate speed', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'positive_growth_center_movement_angle_velocity_mean', 'disp_name': 'growth centroid rotate speed', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_hu1_mean', 'disp_name': 'act hu 1', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_hu4_mean', 'disp_name': 'act hu 4', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_hu5_mean', 'disp_name': 'act hu 5', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_hu6_mean', 'disp_name': 'act hu 6', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_hu7_mean', 'disp_name': 'act hu 7', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_hu8_mean', 'disp_name': 'act hu 8', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_flusser9_mean', 'disp_name': 'act flusser 9', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_flusser10_mean', 'disp_name': 'act flusser 10', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_flusser11_mean', 'disp_name': 'act flusser 11', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_flusser12_mean', 'disp_name': 'act flusser 12', 'format': '{:.3f}'})
    gui_config['statistic_columns'].append({'stat_name': 'activation_flusser13_mean', 'disp_name': 'act flusser 13', 'format': '{:.3f}'})

    gui_config['detail_views'].append({'gui': 'autodisc.gui.ExplorationParametersGUI',
                                       'gui_config': {'run_parameters_config': [{'type': 'table', 'parameters': 'all'},
                                                                                {'type': 'image', 'parameters': [{'name': 'init_state'}], 'gui_config': {'pixel_size': 3}}]
                                                     }})

    gui_config['detail_views'].append({'gui': 'autodisc.gui.ObservationPlayerGUI',
                                      'gui_config': {'pixel_size': 3,
                                                     'elements': [{'type': 'arrow_angle',
                                                                   'is_visible': False,
                                                                   'position': {'source': 'statistics',
                                                                                'name': 'activation_center_position'},
                                                                   'length': {'source': 'statistics',
                                                                              'name': 'activation_center_velocity',
                                                                              'factor': 20},
                                                                   'angle': {'source': 'statistics',
                                                                              'name': 'activation_center_movement_angle'}
                                                                   },
                                                                  {'type': 'arrow_angle',
                                                                   'is_visible': False,
                                                                   'position': {'source': 'statistics',
                                                                                'name': 'positive_growth_center_position'},
                                                                   'length': {'source': 'statistics',
                                                                              'name': 'positive_growth_center_velocity',
                                                                              'factor': 20},
                                                                   'angle': {'source': 'statistics',
                                                                             'name': 'positive_growth_center_movement_angle'}
                                                                   }
                                                                  ]
                                                     }
                                      })

    gui_config['detail_views'].append({'gui': 'autodisc.gui.ObservationPreviewGUI',
                                       'gui_config': {'pixel_size': 3, 'steps': [[0, 10, 20, 30, 40],
                                                                                 [50, 1 / 4, 1 / 2, 3 / 4, -1]]}
                                      })

    gui_config['detail_views'].append({'gui': 'autodisc.gui.StatisticTableGUI',
                                       'gui_config': {'statistics': [{'name': 'activation_mass_mean', 'disp_name': 'activation (mean)', 'format': '{:.3f}'},
                                                                     {'name': 'activation_mass_std', 'disp_name': 'activation (std)', 'format': '{:.3f}'}]
                                                     }})

    gui_config['detail_views'].append({'gui': 'autodisc.gui.StatisticLineGUI',
                                       'gui_config': {'statistics': [{'name': 'activation_mass', 'disp_name': 'activation mass'},
                                                                     {'name': 'positive_growth_mass', 'disp_name': 'positive growth mass'}]
                                                     }})

    gui_config['detail_views'].append({'gui': 'autodisc.gui.StatisticBarGUI',
                                       'gui_config': {'statistics': [{'name': 'activation_mass_mean', 'disp_name': 'activation mass (mean)'},
                                                                     {'name': 'positive_growth_mass_mean', 'disp_name': 'positive growth mass (mean)'}]
                                                     }})

    animal_exploration_gui = ad.gui.ExplorationGUI(explorer=explorer, gui_config=gui_config)
    animal_exploration_gui.run()


if __name__ == '__main__':
    explorer = run_exploration()
    view_exploration_results(explorer)