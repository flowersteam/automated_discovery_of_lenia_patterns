class Config():

    seed = 0
    
    model_type = 'BetaVAE'
    model_init_params = {'n_latents': 8, 'input_size': (256,256), 'beta': 5, 'add_var_to_KLD_loss': bool(0)}
    
    n_epochs = 1400
    train_batch_size = 64
    valid_batch_size = 10
    test_batch_size = 1

    save_output_images = True

    data_augmentation = True

    dataset_filepath = '../data/dataset/dataset.h5'.format(7)

    
    
    
    
    
