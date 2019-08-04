import autodisc as ad
import os
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
plt.ioff()


class PytorchNNRepresentation(ad.core.Representation):

    @staticmethod
    def default_config():
        default_config = ad.core.Representation.default_config()
        default_config.initialization = ad.Config()
        default_config.initialization.type = None # either: 'random_weight', 'load_pretrained_model''
        default_config.save_automatic = True
        default_config.distance_function = None

        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config = config, **kwargs)
        
        if hasattr(self, 'model'):
            print('WARNING: The goal space representation already has a model saved, keeping it and discard new initialization')
            
        else:
            self.load_model(self.config.initialization)
            
    def load_model (self, initialization=None):
        if initialization is None or 'type' not in initialization or initialization.type not in set(['random_weight', 'load_pretrained_model']):
            initialization = ad.Config()
            initialization.type = 'random_weight'
            print('WARNING: wrong goal space initialization given so initializing with default Beta-VAE network')
            
        if initialization.type == 'random_weight':
            print('WARNING: Initializing random weight pytorch network for goal representation')
            # model type: 'AE', 'BetaVAE'
            if 'model_type' in initialization:
                model_type = initialization.model_type
            else:
                model_type = 'BetaVAE'
                print('WARNING: The model type is not specified so initializing Beta-VAE type network')
            try:
                model_cls = getattr(ad.representations.static.pytorchnnrepresentation, model_type)
            except:
                raise ValueError('Unknown initialization.model_type {!r}!'.format(initialization.model_type))
            # model init_params:   
            if 'model_init_params' in initialization:
                model_init_params = initialization.model_init_params
            else:
                model_init_params = {'n_channels': 1, 'n_latents': 8, 'input_size': (256, 256), 'beta': 5.0, 'use_gpu': True}
                print('WARNING: The model init params are not specified so initializing default network with parameters {0}'.format(model_init_params))
            try:
                self.model = model_cls(**model_init_params)
            except:
                raise ValueError('Wrong initialization.model_init_params {!r}!'.format(initialization.model_type)) 
              
        
        elif initialization.type == 'load_pretrained_model':
            print('Initializing pre-trained pytorch network for goal representation')
            if 'load_from_model_path' in initialization:
                if os.path.exists(initialization.load_from_model_path):
                        saved_model = torch.load(initialization.load_from_model_path, map_location='cpu')
                        # model type: 'AE', 'BetaVAE'
                        if 'type' in saved_model:
                            model_type = saved_model['type']
                        else:
                            model_type = 'BetaVAE'
                            print('WARNING: The model type is not specified so initializing Beta-VAE type network')
                        try:
                            model_cls = getattr(ad.representations.static.pytorchnnrepresentation, model_type)
                            self.model_type = model_type
                        except:
                            raise ValueError('Unknown initialization.model_type {!r}!'.format(model_type))
                        # model init_params:   
                        if 'init_params' in saved_model:
                            model_init_params = saved_model['init_params']
                        else:
                            model_init_params = {'n_channels': 1, 'n_latents': 8, 'input_size': (256, 256), 'beta': 5.0, 'use_gpu': True}
                            print('WARNING: The model init params are not specified so initializing default network with parameters {0}'.format(model_init_params))
                        try:
                            self.model = model_cls(**model_init_params)        
                        except:
                            raise ValueError('Wrong initialization model_init_params {!r}!'.format(model_init_params)) 
                        # model state_dict:   
                        try:
                            self.model.load_state_dict(saved_model['state_dict'])
                        except:
                            raise ValueError('Wrong state_dict of the loaded model')
                else:
                    raise ValueError('The model path {0} does not exist: cannot initialize network'.format(initialization.load_from_model_path))
            else:
                raise ValueError('The network cannot be initalized because intialization config does not contain \'load_from_model_path\' parameter')
                          

        # push model on gpu if available
        self.model.eval()
        if self.model.use_gpu:
            self.model = self.model.cuda()
        
        return

            
    def train_epoch (self, train_loader, optimizer):
        self.model.train()
        losses = {}
        for data in train_loader:
            input_img = Variable(data['image'])    
            # forward
            outputs = self.model(input_img)
            batch_losses = self.model.train_loss(outputs, data)
            # backward
            loss = batch_losses['total']
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
    
    
    #def valid_epoch (self, valid_loader, epoch_idx, save_output_images, output_reconstructed_image_folder):
    def valid_epoch (self, valid_loader):
        self.model.eval()
        losses = {}
        with torch.no_grad():
            for data in valid_loader:
                input_img = Variable(data['image'])
                # forward
                outputs = self.model(input_img)
                batch_losses = self.model.valid_losses(outputs, data)
                # save losses
                for k, v in batch_losses.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
                    
        for k, v in losses.items():
            losses [k] = np.mean (v)

        return losses
    
    
    def save_model (self, epoch_idx, optimizer, output_model_filepath):
        init_params = self.model.init_params
        if 'self' in init_params:
                del init_params['self']
        network = {
            'epoch': epoch_idx,
            'type': type(self.model).__name__,
            'init_params': init_params,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
        torch.save(network , output_model_filepath)
        return
    
    
    def valid_epoch_and_save_images(self, valid_loader, images_output_folder):
        self.model.eval()
        losses = {}
        curr_img_idx = 0
        with torch.no_grad():
            for data in valid_loader.dataset:
                data['image'] = data['image'].unsqueeze(0)
                data['label'] = torch.LongTensor([data['label']])
                input_img = Variable(data['image'])
                # forward
                outputs = self.model(input_img)
                loss = self.model.valid_losses(outputs, data)
                # save reconstruction with classification images
                f, axarr = plt.subplots(2, 1)
                f.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
                axarr[0].imshow(input_img.cpu().data[0].squeeze(0).numpy(), cmap='gray')
                axarr[0].axis('off')
                axarr[1].imshow(torch.sigmoid(outputs['recon_x']).cpu().data[0].squeeze(0).numpy(), cmap='gray')
                axarr[1].axis('off')
                if 'recon_y' in outputs:
                    title = 'recon_y: \n {}'.format(outputs['recon_y'].cpu().data[0].numpy())
                else:
                    title = ''
                axarr[1].set_title(title, fontsize=8)
                plt.subplots_adjust(wspace=0.2, hspace=0)
                plt.savefig(os.path.join(images_output_folder, '{}.png'.format(curr_img_idx)))
                plt.close()
                curr_img_idx += 1
                
                # save losses
                for k, v in loss.items():
                    if k not in losses:
                        losses[k] = [v.data.item()]
                    else:
                        losses[k].append(v.data.item())
        
        return losses
    
    
    def save_reconstructed_image(self, input_image, output_filepath):
        self.model.eval()
        with torch.no_grad():
            input_img = Variable(input_image.unsqueeze(0))
            # forward
            outputs = self.model(input_img)
            # save image
            save_image([input_img.cpu().data[0], torch.sigmoid(outputs['recon_x']).cpu().data[0]], output_filepath, nrow = 1, padding=0)
        return
    
    
    def save_trasversal_reconstructed_images_from_image(self, input_image, output_filepath):
        self.model.eval()
        with torch.no_grad():
            input_img = Variable(input_image.unsqueeze(0))
            # encode
            point_mu, point_logvar = self.model.encode(input_img)
            self.save_trasversal_reconstructed_images_from_latent_point(point_mu, output_filepath)   
        return
    
    
    def save_trasversal_reconstructed_images_from_latent_point(self, input_point, output_filepath):
        output_image_list = []
        self.model.eval()
        with torch.no_grad():
            n_rows = int(self.model.n_latents)
            n_cols = 11
            noises = np.linspace(-5,5,n_cols)
            for row_idx in range(n_rows):
                perturbation = torch.zeros_like(input_point)
                for col_idx in range(n_cols):
                    noise = noises[col_idx]
                    perturbation[0, row_idx] = noise
                    point = input_point + perturbation
                    recon_x = torch.sigmoid(self.model.decode(point)).cpu().data[0]
                    output_image_list.append(recon_x)
            save_image(output_image_list, output_filepath, nrow = n_cols, padding=0)
        return


    def calc(self, observations, statistics = None):
        # forward pass of the last state into the network, the output is a vector representation
        last_state = observations['states'][-1]
        last_state = torch.from_numpy(last_state).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            x = Variable(last_state)
            representation = self.model.calc(x).squeeze(0).cpu().numpy()
        return representation


    def calc_distance(self, representation1, representation2):
        if self.config.distance_function is None:
            return super().calc_distance(representation1, representation2)
        else:
            return self.config.distance_function(representation1, representation2, self.config)
