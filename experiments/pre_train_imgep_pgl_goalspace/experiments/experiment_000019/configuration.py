class Config():

    seed = 0
    
    model_type = 'BetaVAE'
    model_init_params = {'n_latents': 8, 'input_size': (256,256), 'beta': 5, 'add_var_to_KLD_loss': bool(1)}
    
    n_epochs = 2000
    train_batch_size = 64
    valid_batch_size = 10


    save_output_images = True

    train_npz_filepath = '../../data/data_{:03d}/train_dataset.npz'.format(6)
    valid_npz_filepath = '../../data/data_{:03d}/valid_dataset.npz'.format(6)
    test_npz_filepath = '../../data/data_{:03d}/test_dataset.npz'.format(6)
    
    
    
    
    
    
