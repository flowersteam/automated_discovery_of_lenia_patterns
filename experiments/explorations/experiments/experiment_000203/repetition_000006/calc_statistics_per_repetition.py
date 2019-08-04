import exputils
import autodisc as ad
import os
import imageio
import numpy as np
import torch
import importlib
from torch.autograd import Variable
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=0)

def collect_final_observation(explorer):

    data = dict()

    for run_data in explorer.data:

        if run_data.observations is not None and len(run_data.observations.states) > 0:

            # rescale values from [0 1] to [0 255] and convert to uint8 for saving as bw image
            img_data = run_data.observations.states[-1] * 255
            img_data = img_data.astype(np.uint8)

            png_image = imageio.imwrite(
                            imageio.RETURN_BYTES,
                            img_data,
                            format='PNG-PIL')

            data['{:06d}.png'.format(run_data.id)] = png_image

    return data


def collect_observations(explorer):

    timestamps = [0, 24, 49, 74, 99, 124, 149, 174, 199]

    data = dict()

    for run_data in explorer.data:

        if run_data.observations is not None and len(run_data.observations.states) > 0:

            for timestamp in timestamps:

                # rescale values from [0 1] to [0 255] and convert to uint8 for saving as bw image
                img_data = run_data.observations.states[timestamp] * 255
                img_data = img_data.astype(np.uint8)

                png_image = imageio.imwrite(
                                imageio.RETURN_BYTES,
                                img_data,
                                format='PNG-PIL')

                data['{:06d}_{:06d}.png'.format(run_data.id, timestamp)] = png_image

    return data


def collect_representation(explorer):
    data = dict()

    data_representations = []

    n_runs = explorer.data.__len__()

    if hasattr(explorer.config.goal_space_representation, 'type') and explorer.config.goal_space_representation.type == 'pytorchnnrepresentation':
        if type(explorer).__name__.lower() == 'goalspaceexplorer':
            explorer_type = 'pretrainVAE'
        elif type(explorer).__name__.lower() == 'onlinelearninggoalexplorer':
            explorer_type = 'onlineVAE'
        model = explorer.goal_space_representation.model
        n_dims_goal_space = model.n_latents
        representation_legend = ['dim {}'.format(dim) for dim in range(n_dims_goal_space)]
    else:
        explorer_type = 'HF'
        model = None
        representation_legend = explorer.config.goal_space_representation.config.statistics
        n_dims_goal_space = len(explorer.config.goal_space_representation.config.statistics)

    for run_data in explorer.data:

        if run_data.observations is not None and len(run_data.observations.states) > 0:
            # fixed representation stored in run_data.reached goal
            if explorer_type == 'HF' or explorer_type == 'pretrainVAE':  #
                data_representations.append(run_data.reached_goal)
            # online version: recompute the reached goal with last trained VAE
            elif explorer_type == 'onlineVAE':
                final_observation = run_data.observations.states[-1]
                input_img = Variable(torch.from_numpy(final_observation).unsqueeze(0).unsqueeze(0).float())
                outputs = model(input_img)
                representation = outputs['mu'].cpu().data.numpy().reshape(n_dims_goal_space)
                data_representations.append(representation)

    data['representation_type'] = explorer_type
    data['n_runs'] = n_runs
    data['n_dims_goal_space'] = n_dims_goal_space
    data['representation_legend'] = representation_legend
    data['coordinates_in_goal_space'] = data_representations
    data['coordinates_in_tsne_space'] = tsne.fit_transform(np.asarray(data_representations))

    return data


# def load_data(experiment_directory):
#
#     dh = ad.ExplorationDataHandler.create(directory=os.path.join(experiment_directory, 'results'))
#     dh.load(load_observations=False, verbose=True)
#
#     dh.config.save_automatic = False
#     dh.config.load_observations = True
#     dh.config.memory_size_observations = 1
#
#     return dh


def load_explorer(experiment_directory):

    # load the full explorer without observations and add its config
    explorer = ad.explorers.GoalSpaceExplorer.load_explorer(os.path.join(experiment_directory, 'results'), run_ids=[], load_observations=False, verbose=False)
    explorer.data.config.load_observations = True
    explorer.data.config.memory_size_observations = 1

    spec = importlib.util.spec_from_file_location('experiment_config', os.path.join(experiment_directory, 'experiment_config.py'))
    experiment_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_config_module)
    explorer.config = experiment_config_module.get_explorer_config()

    return explorer


if __name__ == '__main__':

    experiments = '.'

    statistics = [('final_observation', collect_final_observation, 'zip'),
                  ('observations', collect_observations, 'zip'),
                  ('representations', collect_representation),
                  ]

    exputils.calc_experiment_statistics(statistics, load_explorer, experiments, recalculate_statistics=False, verbose=True)