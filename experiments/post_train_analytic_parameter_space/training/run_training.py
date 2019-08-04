import os
import sys
import time
import numpy as np
import autodisc as ad
from autodisc.representations.static.pytorchnnrepresentation.helper import DatasetHDF5
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import configuration

''' ---------------------------------------------
               PERFORM EPOCH
-------------------------------------------------'''

def train_epoch (train_loader, model, optimizer):
    
    model.train()
    
    losses = {}
    
    for data in train_loader:
        input_img = Variable(data['image'])    
        # forward
        outputs = model(input_img)
        batch_losses = model.train_loss(outputs, data)
        # backward
        loss = batch_losses['total']
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # save losses
        for k, v in batch_losses.items():
            if k not in losses:
                losses[k] = [v.data.item()]
            else:
                losses[k].append(v.data.item())
                
    for k, v in losses.items():
        losses [k] = np.mean (v)
    
    return losses


def valid_epoch (epoch, valid_loader, model, save_output_images, output_valid_reconstruction_folder):
    model.eval()
    losses = {}
    with torch.no_grad():
        for data in valid_loader:
            input_img = Variable(data['image'])
            # forward
            outputs = model(input_img)
            batch_losses = model.valid_losses(outputs, data)
            # save losses
            for k, v in batch_losses.items():
                if k not in losses:
                    losses[k] = [v.data.item()]
                else:
                    losses[k].append(v.data.item())
                
    for k, v in losses.items():
        losses [k] = np.mean (v)
    
    # save reconstructed images versus original images for last batch
    if save_output_images and epoch % 50 == 0:
        input_images = input_img.cpu().data
        output_images = torch.sigmoid(outputs['recon_x']).cpu().data
        n_images = data['image'].size()[0]
        vizu_tensor_list = [None] * (2*n_images)
        vizu_tensor_list[:n_images] = [input_images[n] for n in range(n_images)]
        vizu_tensor_list[n_images:] = [output_images[n] for n in range(n_images)]
        filename = os.path.join (output_valid_reconstruction_folder, 'Epoch{0}.png'.format(epoch))
        save_image(vizu_tensor_list, filename, nrow=n_images, padding=0)
        
    
    return losses
   
   
'''
-------------------------------------------------
               TRAINING LOOP
-------------------------------------------------
'''

def train():
        
    # configuration file
    print("Loading the configuration ... \n")
    config = configuration.Config()    
    
    # training parameters
    model_type = config.model_type
    model_init_params = config.model_init_params
    img_size = model_init_params['input_size']
    n_epochs = config.n_epochs
    
    save_output_images = config.save_output_images

    # set seed
    np.random.seed(config.seed)
    
    # load datasets 
    print("Loading the datasets ... \n")
    train_dataset = DatasetHDF5(filepath=config.dataset_filepath,
                                split='train',
                                img_size=img_size,
                                data_augmentation = config.data_augmentation)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.train_batch_size,
                              shuffle=True)

    valid_dataset = DatasetHDF5(filepath=config.dataset_filepath,
                                split='valid',
                                img_size = img_size)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.valid_batch_size,
                              shuffle=True)

    print("Loading the model ... \n")
    model_cls = getattr(ad.representations.static.pytorchnnrepresentation, model_type)
    model = model_cls (**model_init_params)
        
    if model.use_gpu:
        model = model.cuda()
        
    load_weight = False
    weight_to_load_filename = ''
    if load_weight:
        print ("=> Loading saved model {0}".format(weight_to_load_filename))
        model.load_state_dict(torch.load(weight_to_load_filename))

    # optimizer
    learning_rate = 1e-3
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # output files
    output_training_folder = 'training'
    if os.path.exists(output_training_folder):
        print('WARNING: training folder already exists')
    else:
        os.makedirs(output_training_folder)
    output_valid_reconstruction_folder = os.path.join(output_training_folder, 'reconstruction_images')
    if not os.path.exists(output_valid_reconstruction_folder):
        os.makedirs(output_valid_reconstruction_folder)
    output_models_folder = os.path.join(output_training_folder, 'models')
    if not os.path.exists(output_models_folder):
        os.makedirs(output_models_folder)

    train_filepath = os.path.join(output_training_folder, 'loss_train.cvs')
    train_file = open (train_filepath, 'a')
    train_file.write("Epoch\tloss\n")
    train_file.close()
    
    valid_filepath = os.path.join(output_training_folder, 'loss_valid.cvs')
    valid_file = open (valid_filepath, 'a')
    valid_file.write("Epoch\ttotal\tBCE\tKLD\tKLD_var\n")
    valid_file.close()

    # training loop
    best_valid_loss = sys.float_info.max
    print(" Start training ... \n")
    
    for epoch in range(n_epochs):
        
        # training
        tstart0 = time.time()
        train_losses = train_epoch (train_loader, model, optimizer)
        tend0 = time.time()
        
        train_file = open ( os.path.join(output_training_folder, 'loss_train.cvs'), 'a')
        train_file.write("Epoch: {0}".format(epoch))
        for k, v in train_losses.items():
            train_file.write("\t{0}: {1:.6f}".format(k,v))
        train_file.write("\n")
        train_file.close()
        
        # validation
        tstart1 = time.time()
        valid_losses = valid_epoch (epoch, valid_loader, model, save_output_images, output_valid_reconstruction_folder)
        tend1 = time.time()
        
        valid_file = open ( os.path.join(output_training_folder, 'loss_valid.cvs'), 'a')
        valid_file.write("Epoch: {0}".format(epoch))
        for k, v in valid_losses.items():
            valid_file.write("\t{0}: {1:.6f}".format(k,v))
        valid_file.write("\n")
        valid_file.close()
        
        # print summary
        print("Epoch {0}: train loss {1:.6f} (time: {2} secs), valid loss {3:.6f} (time: {4} secs)\n".format(epoch, train_losses['total'], tend0-tstart0, valid_losses['total'], tend1-tstart1))
        
        model_name = type(model).__name__
        model_init_params = model.init_params
        if 'self' in model_init_params:
                del model_init_params['self']
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        # save current epoch weight file with optimizer if we want to relaunch training from that point
        network = {
        'epoch': epoch,
        'type': model_name,
        'init_params': model_init_params,
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
        }
        torch.save(network , os.path.join (output_models_folder, 'current_weight_model.pth'))
        
        # save the best weights on the valid set for further inference
        valid_loss = valid_losses['total']
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            network = {
            'epoch': epoch,
            'type': model_name,
            'init_params': model_init_params,
            'state_dict': model_state_dict,
        }
        torch.save(network , os.path.join (output_models_folder, 'best_weight_model.pth'))

    #  close dataset files
    train_dataset.close()
    valid_dataset.close()
        

if __name__ == "__main__":
    train()
