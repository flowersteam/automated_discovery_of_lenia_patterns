import os
import numpy as np


dataset_1_path = '../data_004'
train_data_1 = np.load(os.path.join(dataset_1_path, 'train_dataset.npz'))
valid_data_1 = np.load(os.path.join(dataset_1_path, 'valid_dataset.npz'))
test_data_1 = np.load(os.path.join(dataset_1_path, 'test_dataset.npz'))


dataset_2_path = '../data_005'
train_data_2 = np.load(os.path.join(dataset_2_path, 'train_dataset.npz'))
valid_data_2 = np.load(os.path.join(dataset_2_path, 'valid_dataset.npz'))
test_data_2 = np.load(os.path.join(dataset_2_path, 'test_dataset.npz'))

assert train_data_1['observations'].shape == train_data_2['observations'].shape
assert valid_data_1['observations'].shape == valid_data_2['observations'].shape
assert test_data_1['observations'].shape == test_data_2['observations'].shape


dataset_3_path = '.'
    
train_images = np.concatenate((train_data_1['observations'], train_data_2['observations']), axis = 0)
valid_images = np.concatenate((valid_data_1['observations'], valid_data_2['observations']), axis = 0)
test_images = np.concatenate((test_data_1['observations'], test_data_2['observations']), axis = 0)

train_labels = np.concatenate((train_data_1['labels'], train_data_2['labels']), axis = 0)
valid_labels = np.concatenate((valid_data_1['labels'], valid_data_2['labels']), axis = 0)
test_labels = np.concatenate((test_data_1['labels'], test_data_2['labels']), axis = 0)


np.savez(os.path.join(dataset_3_path, 'train_dataset.npz'), observations = train_images, labels = train_labels)
np.savez(os.path.join(dataset_3_path, 'valid_dataset.npz'), observations = valid_images, labels = valid_labels)
np.savez(os.path.join(dataset_3_path, 'test_dataset.npz'), observations = test_images, labels = test_labels)


with open(os.path.join (dataset_3_path, 'dataset_summary.csv'), 'w') as f:
    f.write('n_tot\t{}\n'.format(train_images.shape[0] + valid_images.shape[0] + test_images.shape[0]))
    f.write('n_train\t{}\n'.format(train_images.shape[0]))
    f.write('n_valid\t{}\n'.format(valid_images.shape[0]))
    f.write('n_test\t{}\n'.format(test_images.shape[0]))
