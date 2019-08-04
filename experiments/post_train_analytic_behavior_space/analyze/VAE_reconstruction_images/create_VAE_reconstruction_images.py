import os
import numpy as np
import autodisc as ad
from autodisc.representations.static.pytorchnnrepresentation.helper import DatasetHDF5
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from autodisc.gui.jupyter.misc import create_colormap, transform_image_from_colormap
from PIL import Image


# INPUT INFO
experiment_id = 171102
dataset_id = 7
n_max_images = 300

# OUTPUT INFO
output_image_folder = './images_colored/'
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)
img_filename_template = output_image_folder + 'pattern_{:06d}_{}'
img_suffix = '.png'
colored = True
colormap = create_colormap(np.array([[255,255,255], [119,255,255],[23,223,252],[0,190,250],[0,158,249],[0,142,249],[81,125,248],[150,109,248],[192,77,247],[232,47,247],[255,9,247],[200,0,84]])/255*8)



# LOAD MODEL    
model_path = '../../experiments/experiment_{:06d}/training/models/best_weight_model.pth'.format(experiment_id)
#model_path = '../../prior_data/pretrained_representation/representation_000118/best_weight_model.pth'
#model_path = '../../experiments/experiment_{:06d}/repetition_{:06d}/trained_representation/saved_models/stage_{:06d}_weight_model.pth'.format(experiment_id, repetition_id, stage_id)
print("Loading the trained model ... \n")
if os.path.exists(model_path):
        saved_model = torch.load (model_path, map_location='cpu')
        model_cls = getattr(ad.representations.static.pytorchnnrepresentation, saved_model['type'])
        if 'self' in saved_model['init_params']:
                del saved_model['init_params']['self']
        model = model_cls (**saved_model['init_params'])        
        model.load_state_dict(saved_model['state_dict'])
        model.eval()
        model.use_gpu = False
else:
    raise ValueError('The model {!r} does not exist!'.format(model_path))
input_size = model.input_size

# LOAD DATASET
test_filepath = '../../data/data_{:03d}/dataset/dataset.h5'.format(dataset_id)
#test_npz_filepath = '../../../representation_pretrain/data/data_006/valid_dataset.npz'
#test_npz_filepath = '../../experiments/experiment_{:06d}/repetition_{:06d}/trained_representation/stages_summary/stage_{:06d}/valid_dataset.npz'.format(experiment_id, repetition_id, stage_id)
print("Loading the test dataset ... \n")
if os.path.exists(test_filepath):

    test_dataset = DatasetHDF5(filepath=test_filepath,
                                split='test',
                                img_size = input_size)
    test_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=True)

#    test_dataset_npz = np.load(test_npz_filepath)
#    test_batch_size = 1
#    test_dataset = DatasetHDF5(input_size)
#    if 'images' in test_dataset_npz:
#        test_dataset.update(test_dataset_npz['images'].shape[0], torch.from_numpy(test_dataset_npz['images']).float(), test_dataset_npz['labels'])
#    elif 'observations' in test_dataset_npz:
#        test_dataset.update(test_dataset_npz['observations'].shape[0], torch.from_numpy(test_dataset_npz['observations']).float(), test_dataset_npz['labels'])
#    else:
#        raise ValueError('dataset not properly defined')
#    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
else:
    raise ValueError('The dataset {!r} does not exist!'.format(test_npz_filepath))

# LOOP OVER DATA 
print("Testing the images ... \n")
with torch.no_grad():
    img_idx = 0

    for data in test_loader:
        # input
        x = Variable(data['image'])
        output = model(x)
        recon_x = torch.sigmoid(output['recon_x'])
        
        # convert to array
        x = x.cpu().data.numpy().reshape((input_size[0], input_size[1]))
        recon_x = recon_x.cpu().data.numpy().reshape((input_size[0], input_size[1]))
        
        # convert to image
        x_PIL = Image.fromarray(np.uint8(x*255.0))
        recon_x_PIL = Image.fromarray(np.uint8(recon_x*255.0))
            
        # convert to color image
        if colored:
            x_PIL = transform_image_from_colormap(x_PIL, colormap).convert('RGBA')
            recon_x_PIL = transform_image_from_colormap(recon_x_PIL, colormap).convert('RGBA') 
        
        # save image
        x_PIL.save(img_filename_template.format(img_idx, 'original') + img_suffix)
        recon_x_PIL.save(img_filename_template.format(img_idx, 'reconstructed') + img_suffix)

        img_idx += 1

        if img_idx >= n_max_images:
                break;




        
        
    
