import os
import numpy as np

import configuration

import autodisc as ad
from autodisc.representations.static.pytorchnnrepresentation.helper import Dataset

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import importlib
from autodisc.representations.static.pytorchnnrepresentation import helper 

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA

import torch.nn.functional as F    


''' ---------------------------------------------------------------------------------
        FORWARD PASS OF TEST DATASET THROUGH TRAINED MODEL + LOSS COMPUTATION
----------------------------------------------------------------------------------'''
def test():
    # configuration file
    print("Loading the configuration ... \n")
    config = configuration.Config()
    model_init_params = config.model_init_params
    img_size = model_init_params['input_size']
    
    # load datasets 
    print("Loading the test dataset ... \n")
    test_npz_filepath = config.test_npz_filepath
    test_dataset_npz = np.load(test_npz_filepath)
    test_batch_size = 1
    test_dataset = Dataset(img_size)
    test_dataset.update(test_dataset_npz['observations'].shape[0], torch.from_numpy(test_dataset_npz['observations']).float(), test_dataset_npz['labels'])
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    
    # load the trained model
    print("Loading the trained model ... \n")
    model_path = 'training/models/best_weight_model.pth'
    if os.path.exists(model_path):
            saved_model = torch.load (model_path, map_location='cpu')
            model_cls = getattr(ad.representations.static.pytorchnnrepresentation, saved_model['type'])
            model = model_cls (**saved_model['init_params'])        
            model.load_state_dict(saved_model['state_dict'])
            model.eval()
            model.use_gpu = False
    else:
        raise ValueError('The model {!r} does not exist!'.format(model_path))
        
    # output files
    output_testing_folder = 'testing'
    if os.path.exists(output_testing_folder):
        print('WARNING: testing folder already exists')
    else:
        os.makedirs(output_testing_folder)
    
    output_test_dataset_filename = os.path.join(output_testing_folder, 'output_test_dataset.npz')
    if os.path.exists(output_test_dataset_filename):
        print('WARNING: the output test dataset already exists, skipping the forward pass on test dataset')
        return
    
    
    # prepare output arrays
    input_size = saved_model['init_params']['input_size']
    n_latents = saved_model['init_params']['n_latents']
    test_x = np.empty((test_loader.__len__(), input_size[0], input_size[1]))
    test_recon_x = np.empty((test_loader.__len__(), input_size[0], input_size[1]))
    test_y = np.empty(test_loader.__len__())
    test_recon_y = np.empty((test_loader.__len__(), 3))
    test_mu = np.empty((test_loader.__len__(), n_latents))
    test_sigma = np.empty((test_loader.__len__(), n_latents)) # FOR AE: mu=z, sigma=0
    test_MSE_loss = np.empty(test_loader.__len__())
    test_BCE_loss = np.empty(test_loader.__len__())
    test_KLD_loss = np.empty(test_loader.__len__())
    test_KLD_loss_per_latent_dim = np.empty((test_loader.__len__(), n_latents))
    test_KLD_loss_var = np.empty(test_loader.__len__())
    test_CE_loss = np.empty(test_loader.__len__())
    
    
    # Loop over the test images:
    print("Testing the images ... \n")
    with torch.no_grad():
        idx = 0
    
        for data in test_loader:
            # input
            input_img = Variable(data['image'])
            input_label = Variable(data['label'])
            test_x[idx,:,:] = input_img.cpu().data.numpy().reshape((input_size[0], input_size[1]))
            test_y[idx] = input_label.cpu().data.numpy()
            # forward pass outputs
            outputs = model(input_img)
            
            test_mu[idx,:] = outputs['mu'].cpu().data.numpy().reshape(n_latents)
            test_sigma[idx,:] = outputs['logvar'].exp().sqrt().cpu().data.numpy().reshape(n_latents)
            test_recon_x[idx,:,:] = torch.sigmoid(outputs['recon_x']).cpu().data.numpy().reshape((input_size[0], input_size[1]))
            # compute reconstruction losses
            MSE_loss = helper.MSE_loss(torch.sigmoid(outputs['recon_x']), input_img).cpu().data.numpy().reshape(1)
            BCE_loss = helper.BCE_with_digits_loss(outputs['recon_x'], input_img).cpu().data.numpy().reshape(1)
            
            if 'recon_y' in outputs:
                test_recon_y[idx,:] = F.softmax(outputs['recon_y']).cpu().data.numpy().reshape(3)
                CE_loss = (F.cross_entropy(outputs['recon_y'], input_label, size_average=False) / input_label.size()[0])
            else:
                test_recon_y[idx,:] = np.zeros(3)
                CE_loss = 0
                
            
            KLD_loss, KLD_loss_per_latent_dim, KLD_loss_var = helper.KLD_loss(outputs['mu'], outputs['logvar'])
            test_MSE_loss[idx] = MSE_loss
            test_BCE_loss[idx] = BCE_loss
            test_CE_loss[idx] = CE_loss
            test_KLD_loss[idx] = KLD_loss.cpu().data.numpy().reshape(1)
            test_KLD_loss_per_latent_dim[idx,:] = KLD_loss_per_latent_dim.cpu().data.numpy().reshape(n_latents)
            test_KLD_loss_var[idx] = KLD_loss_var.cpu().data.numpy().reshape(1)
    
            idx += 1
        
        
    # Save in the experiment test folder
    print("Saving the results ... \n")
    np.savez(output_test_dataset_filename, x = test_x, recon_x = test_recon_x, y = test_y, recon_y = test_recon_y, mu = test_mu, sigma = test_sigma, MSE_loss = test_MSE_loss, BCE_loss = test_BCE_loss, KLD_loss = test_KLD_loss, KLD_loss_per_latent_dim = test_KLD_loss_per_latent_dim, KLD_loss_var = test_KLD_loss_var, CE_loss = test_CE_loss)
    return


''' ---------------------------------------------------------------------------------
        WORST RECONSTRUCTION CASES
----------------------------------------------------------------------------------'''
def show_worst_reconstruction_cases():
    # load the tested dataset
    data = np.load('testing/output_test_dataset.npz')
    x = data['x']
    recon_x = data['recon_x']
    MSE_loss = data['MSE_loss']
    BCE_loss = data['BCE_loss']
    n_test_images = x.shape[0]
    
    # create output folder
    output_folder = 'testing/reconstruction_worst_cases'
    if os.path.exists(output_folder):
        print('WARNING: the worst reconstructed cases folder already exists, skipping the saving of worst cases on test dataset')
        return
    else:
        os.makedirs(output_folder)
   
    # MSE loss
    ## normalize and sort
    normalized_MSE_loss = np.asarray([MSE_loss[idx]/max(np.sum(x[idx]>0.),1.0) for idx in range(n_test_images)])
    indexes_sorted_by_normalized_MSE_loss = np.argsort(normalized_MSE_loss).astype('int')

    # plot and save the ten worst cases
    f, axarr = plt.subplots(2, 10, figsize=(10,3))
    for i in range(10):
        axarr[0,i].imshow(x[indexes_sorted_by_normalized_MSE_loss[-(i+1)]], cmap='gray')
        axarr[0,i].axis('off')
        axarr[0,i].set_title('{:.01f}'.format(normalized_MSE_loss[indexes_sorted_by_normalized_MSE_loss[-(i+1)]]))
        axarr[1,i].imshow(recon_x[indexes_sorted_by_normalized_MSE_loss[-(i+1)]], cmap='gray')
        axarr[1,i].axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(os.path.join(output_folder,'MSE_worst_cases.png'))
    plt.close()
    
    # BCE loss
    ## normalize and sort
    normalized_BCE_loss = np.asarray([BCE_loss[idx]/max(np.sum(x[idx]>0.),1.0) for idx in range(n_test_images)])
    indexes_sorted_by_normalized_BCE_loss = np.argsort(normalized_BCE_loss).astype('int')

    # plot and save the ten worst cases
    f, axarr = plt.subplots(2, 10, figsize=(10,3))
    for i in range(10):
        axarr[0,i].imshow(x[indexes_sorted_by_normalized_BCE_loss[-(i+1)]], cmap='gray')
        axarr[0,i].axis('off')
        axarr[0,i].set_title('{:.01f}'.format(normalized_BCE_loss[indexes_sorted_by_normalized_BCE_loss[-(i+1)]]))
        axarr[1,i].imshow(recon_x[indexes_sorted_by_normalized_BCE_loss[-(i+1)]], cmap='gray')
        axarr[1,i].axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(os.path.join(output_folder,'BCE_worst_cases.png'))
    plt.close()
    return

def show_worst_classification_cases():
    np.set_printoptions(2,suppress=True)
    # load the tested dataset
    data = np.load('testing/output_test_dataset.npz')
    x = data['x']
    recon_x = data['recon_x']
    y = data['y']
    recon_y = data['recon_y']
    CE_loss = data['CE_loss']
    n_classes = recon_y.shape[1]
    
    # create output folder
    output_folder = 'testing/classification_worst_cases'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for class_id in range(n_classes):
        class_ids = np.where(y == class_id)
        class_CE_loss = CE_loss[class_ids]
        ## normalize and sort
        indexes_sorted_by_normalized_CE_loss = class_ids[0][np.argsort(class_CE_loss)]
    
        # plot and save the ten worst cases
        f, axarr = plt.subplots(2, 10, figsize=(10,3))
        f.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        for i in range(10):
            axarr[0,i].imshow(x[indexes_sorted_by_normalized_CE_loss[-(i+1)]], cmap='gray')
            axarr[0,i].axis('off')
            axarr[1,i].imshow(recon_x[indexes_sorted_by_normalized_CE_loss[-(i+1)]], cmap='gray')
            axarr[1,i].axis('off')
            axarr[1,i].set_title('loss: {:.01f} \n recon_y: \n {}'.format(CE_loss[indexes_sorted_by_normalized_CE_loss[-(i+1)]], recon_y[indexes_sorted_by_normalized_CE_loss[-(i+1)]]), fontsize=8)
    
        plt.subplots_adjust(wspace=0.2, hspace=0)
        plt.savefig(os.path.join(output_folder, 'class_{:02d}_CE_worst_cases.png'.format(class_id)))
        plt.close()
    
    return


''' ---------------------------------------------------------------------------------
        LOSS STATISTICS
----------------------------------------------------------------------------------'''
def compute_statistics():
    # load the losses of tested dataset
    data = np.load('testing/output_test_dataset.npz')
    MSE_loss = data['MSE_loss']
    BCE_loss = data['BCE_loss']
    KLD_loss = data['KLD_loss']
    KLD_loss_per_latent_dim = data['KLD_loss_per_latent_dim']
    KLD_loss_var = data['KLD_loss_per_latent_dim']
    CE_loss = data['CE_loss']
    
    # create output statistics file
    output_file = open('testing/statistics.cvs', 'w')
    output_file.write('\n\n')
    output_file.write('  \t Mean \t Med \t Std \t Min \t Max \t 90-percentile \n')
    output_file.write(' MSE \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(np.mean(MSE_loss), np.median(MSE_loss), np.std(MSE_loss), np.min(MSE_loss), np.max(MSE_loss), np.percentile(MSE_loss, 90) ))
    output_file.write(' BCE \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(np.mean(BCE_loss), np.median(BCE_loss), np.std(BCE_loss), np.min(BCE_loss), np.max(BCE_loss), np.percentile(BCE_loss, 90) ))
    output_file.write(' KLD \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(np.mean(KLD_loss), np.median(KLD_loss), np.std(KLD_loss), np.min(KLD_loss), np.max(KLD_loss), np.percentile(KLD_loss, 90) ))
    output_file.write(' KLD_var \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(np.mean(KLD_loss_var), np.median(KLD_loss_var), np.std(KLD_loss_var), np.min(KLD_loss_var), np.max(KLD_loss_var), np.percentile(KLD_loss_var, 90) ))
    output_file.write(' CE \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(np.mean(CE_loss), np.median(CE_loss), np.std(CE_loss), np.min(CE_loss), np.max(CE_loss), np.percentile(CE_loss, 90) ))

    output_file.write('\n\n')
    
    output_file.write(' Number of used dimensions: {:03d}'.format(np.sum(KLD_loss_per_latent_dim.mean(axis=0) > 0.1)))
    output_file.write(' Sorted dimensions per decreasing KLD (averadged on the dataset): {}'.format(np.argsort(KLD_loss_per_latent_dim.mean(axis=0)).astype('int')))
    return


''' ---------------------------------------------------------------------------------
        LATENT SPACE ANALYSIS
----------------------------------------------------------------------------------'''
def latent_space_projections_visualisation():
    # load the tested dataset
    data = np.load('testing/output_test_dataset.npz')
    mu = data['mu']
    sigma = data['sigma']
    n_latents = mu.shape[1]
    
    # create output folder
    output_folder = 'testing/latent_space_analysis'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, 'latent_space_projections.png')
    if os.path.exists(output_filename):
        print('WARNING: the projected latent space visualisation already exists')
        #return
    
    # Plot latent space 2 by 2
    n_total_plots = int((n_latents-1)*(n_latents)/2)
    n_rows = int(round(np.sqrt(n_total_plots)))
    n_cols = int(n_rows)
    if n_total_plots % pow(n_rows,2) > 0:
        n_cols += 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize = (2*n_rows, 2*n_cols), subplot_kw={'aspect': 'equal'})    
    limit_plot = np.max(np.abs(mu))
    
    idx = 0
    for dim1 in range(n_latents):
        for dim2 in range(dim1+1, n_latents):
            i,j = np.unravel_index(idx, (n_rows,n_cols))
            ells0 = [Ellipse(xy=[mu[i,dim1], mu[i,dim2]], width =sigma[i,dim1], height = sigma[i,dim2]) for i in range(mu.shape[0])]
            
            if (n_rows==1 and n_cols==1):
                for e in ells0:
                    ax.add_artist(e)
                    e.set_clip_box(ax.bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((0., 0., 1.))
                ax.set_xlim(-limit_plot,limit_plot)
                ax.set_ylim(-limit_plot,limit_plot)
                ax.set_xlabel('dim{0}'.format(dim1))
                ax.set_ylabel('dim{0}'.format(dim2))
            else:
                for e in ells0:
                    ax[i,j].add_artist(e)
                    e.set_clip_box(ax[i,j].bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((0., 0., 1.))
                ax[i,j].set_xlim(-limit_plot,limit_plot)
                ax[i,j].set_ylim(-limit_plot,limit_plot)
                ax[i,j].set_xlabel('dim{0}'.format(dim1))
                ax[i,j].set_ylabel('dim{0}'.format(dim2))

            idx += 1
    
    fig.tight_layout()       
    plt.savefig(output_filename)
    plt.close()
    return

def latent_space_projections_visualisation_2():
    # load the tested dataset
    data = np.load('testing/output_test_dataset.npz')
    mu = data['mu']
    sigma = data['sigma']
    y = data['y']
    animal_ids = np.where(y==0)[0]
    divergent_structure_ids = np.where(y==1)[0]
    non_animal_ids = np.where(y==2)[0]
    n_latents = mu.shape[1]
    
    # create output folder
    output_folder = 'testing/latent_space_analysis'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, 'latent_space_projections.png')
    if os.path.exists(output_filename):
        print('WARNING: the projected latent space visualisation already exists, skipping it')
        return
    
    # Plot latent space 2 by 2
    n_total_plots = int((n_latents-1)*(n_latents)/2)
    n_rows = int(round(np.sqrt(n_total_plots)))
    n_cols = int(n_rows)
    if n_total_plots % pow(n_rows,2) > 0:
        n_cols += 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize = (2*n_rows, 2*n_cols), subplot_kw={'aspect': 'equal'})    
    limit_plot = np.max(np.abs(mu))
    
    idx = 0
    for dim1 in range(n_latents):
        for dim2 in range(dim1+1, n_latents):
            i,j = np.unravel_index(idx, (n_rows,n_cols))
            ells0 = [Ellipse(xy=[mu[i,dim1], mu[i,dim2]], width = max(sigma[i,dim1], limit_plot/50), height = max(sigma[i,dim2],  limit_plot/50)) for i in animal_ids]
            ells1 = [Ellipse(xy=[mu[i,dim1], mu[i,dim2]], width = max(sigma[i,dim1], limit_plot/50), height = max(sigma[i,dim2],  limit_plot/50)) for i in divergent_structure_ids]
            ells2 = [Ellipse(xy=[mu[i,dim1], mu[i,dim2]], width = max(sigma[i,dim1], limit_plot/50), height = max(sigma[i,dim2],  limit_plot/50)) for i in non_animal_ids]
            
            if (n_rows==1 and n_cols==1):
                for e in ells0:
                    ax.add_artist(e)
                    e.set_clip_box(ax.bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((1., 0., 0.))
                for e in ells1:
                    ax.add_artist(e)
                    e.set_clip_box(ax.bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((0., 1., 0.))
                for e in ells2:
                    ax.add_artist(e)
                    e.set_clip_box(ax.bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((0., 0., 1.))
                ax.set_xlim(-limit_plot,limit_plot)
                ax.set_ylim(-limit_plot,limit_plot)
                ax.set_xlabel('dim{0}'.format(dim1))
                ax.set_ylabel('dim{0}'.format(dim2))
            else:
                for e in ells0:
                    ax[i,j].add_artist(e)
                    e.set_clip_box(ax[i,j].bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((1., 0., 0.))
                for e in ells1:
                    ax[i,j].add_artist(e)
                    e.set_clip_box(ax[i,j].bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((0., 1., 0.))
                for e in ells2:
                    ax[i,j].add_artist(e)
                    e.set_clip_box(ax[i,j].bbox)
                    e.set_alpha(0.9)
                    e.set_facecolor((0., 0., 1.))
                ax[i,j].set_xlim(-limit_plot,limit_plot)
                ax[i,j].set_ylim(-limit_plot,limit_plot)
                ax[i,j].set_xlabel('dim{0}'.format(dim1))
                ax[i,j].set_ylabel('dim{0}'.format(dim2))

            idx += 1
    
    fig.tight_layout()       
    plt.savefig(output_filename)
    plt.close()
    return


def latent_space_PCA_analysis():
    # load the tested dataset
    data = np.load('testing/output_test_dataset.npz')
    mu = data['mu']
    n_latents = mu.shape[1]
    
    # create output folder
    output_folder = 'testing/latent_space_analysis'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, 'latent_space_explained_variance_by_PCA.png')
    if os.path.exists(output_filename):
        print('WARNING: the latent space pca analysis already exists, skipping it')
        return

    # do pca 
    pca = PCA(n_components=n_latents)
    pca.fit(mu)
    eig_vals = pca.explained_variance_
    eig_vecs = pca.components_
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.figure(figsize=(20,20))
    plt.bar(x=[i for i in range(1,n_latents+1)], height=var_exp, width=.8, label='Individual')
    plt.scatter(x=[i for i in range(1,n_latents+1)], y=cum_var_exp, linestyle='-', marker='o', label='Individual')
    plt.xlabel('principal component')
    plt.ylabel('Explained variance in percent')
    eig_vecs_text = ''
    i = 0
    for eig_vec in eig_vecs:
        if var_exp[i]<1:
            eig_vecs_text += 'PC {0}: {1}\n'.format(i, np.array2string(eig_vec, precision=1, separator=',', suppress_small=True))
        i += 1
    plt.text(n_latents,100, eig_vecs_text, horizontalalignment='right', verticalalignment='top')
    plt.legend()
    plt.savefig(output_filename)
    plt.close()
    
    return


def latent_space_traversal_analysis(n_points):    
    # load the trained model
    model_path = 'training/models/best_weight_model.pth'
    if os.path.exists(model_path):
            saved_model = torch.load (model_path, map_location='cpu')
            model_cls = getattr(ad.representations.static.pytorchnnrepresentation, saved_model['type'])
            model = model_cls (**saved_model['init_params'])        
            model.load_state_dict(saved_model['state_dict'])
            model.eval()
            model.use_gpu = False
    else:
        raise ValueError('The model {!r} does not exist!'.format(model_path))
    
    # create output folder
    output_parent_folder = 'testing/latent_space_analysis'
    if not os.path.exists(output_parent_folder):
        os.makedirs(output_parent_folder)
    output_folder = os.path.join(output_parent_folder, 'traversal_analysis')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # sample n random points in the latent space
    n_latents = saved_model['init_params']['n_latents']
    input_size = saved_model['init_params']['input_size']
    points = torch.from_numpy(np.random.randn(n_points, n_latents)).float()
   
    
    for point_id in range(n_points):
        n_rows = int(n_latents)
        n_cols = 11
        noises = np.linspace(-5,5,n_cols)
        fig, ax = plt.subplots(n_rows, n_cols)
        fig.subplots_adjust(0, 0.01, 1, 0.96, 0.2, 0.3)
        for row_idx in range(n_latents):
            perturbation = torch.zeros_like(points[point_id,:])
            for col_idx in range(n_cols):
                noise = noises[col_idx]
                perturbation[row_idx] = noise
                input_point = (points[point_id,:] + perturbation).unsqueeze(0)
                recon_x = torch.sigmoid(model.decoder(input_point)).cpu().data.numpy().reshape((input_size[0], input_size[1]))
                if hasattr(model, 'fc'):
                    recon_y = F.softmax(model.fc(input_point)).cpu().data.numpy().reshape(3).argmax()
                    title = str(recon_y)
                else:
                    title = ''
                ax[row_idx,col_idx].imshow(recon_x, cmap = 'gray')
                ax[row_idx,col_idx].set_title(title, pad=0)
                ax[row_idx,col_idx].axis('off')
    
        plt.savefig(os.path.join(output_folder, 'point_{}.png'.format(point_id)))
        plt.close()
    
    return

''' ---------------------------------------------------------------------------------
        ANALYSE TRAINING
----------------------------------------------------------------------------------'''
def analyse_training():
    # plot train and valid loss curves
    train_losses = {}
    n_epochs = 0
    with open(os.path.join('training', 'loss_train.cvs'), 'r') as f:
        lineslist = [line.rstrip() for line in f]
        for line in lineslist:
            line = line.split('\t')
            if line[0][:6] == 'Epoch:':
                for col in range(1, len(line)):
                    k,v = line[col].split(' ')
                    k = k[:-1]
                    if k not in train_losses:
                            train_losses[k] = [float(v)]
                    else:
                        train_losses[k].append(float(v))
                n_epochs +=1
            
    valid_losses = {}
    with open(os.path.join('training', 'loss_valid.cvs'), 'r') as f:
        lineslist = [line.rstrip() for line in f]
        for line in lineslist:
            line = line.split('\t')
            if line[0][:6] == 'Epoch:':
                for col in range(1, len(line)):
                    k,v = line[col].split(' ')
                    k = k[:-1]
                    if k not in valid_losses:
                            valid_losses[k] = [float(v)]
                    else:
                        valid_losses[k].append(float(v))
    for key in train_losses:
        output_filename = os.path.join('training', '{}_curves.png'.format(key))
        fig, ax = plt.subplots()
        ax.plot(range(n_epochs), train_losses[key], 'k', color='red', label='train data')
        ax.plot(range(n_epochs), valid_losses[key], 'k', color='green', label='valid data')
        ax.set_xlabel('train epoch')
        ax.set_ylabel('{}'.format(key))
        ax.legend(loc='upper center', shadow=True)
        plt.savefig(output_filename)
        plt.close()
        
   
    
    return

''' ---------------------------------------------------------------------------------
        MAIN
----------------------------------------------------------------------------------'''
if __name__ == "__main__":
    analyse_training()
    test()
    show_worst_reconstruction_cases()
    #show_worst_classification_cases()
    compute_statistics()
    latent_space_projections_visualisation()
    latent_space_PCA_analysis()
    latent_space_traversal_analysis(20)

    

