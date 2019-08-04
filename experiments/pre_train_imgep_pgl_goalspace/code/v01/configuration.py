class Config():

    seed = 0
    
    model_type = '<model_type>'
    model_init_params = {'n_latents': 8, 'input_size': (256,256), 'beta': <beta>, 'add_var_to_KLD_loss': bool(<add_var_to_KLD_loss>)}
    
    n_epochs = 2000
    train_batch_size = 64
    valid_batch_size = 10


    save_output_images = True

    train_npz_filepath = '../../data/data_{:03d}/train_dataset.npz'.format(<dataset_id>)
    valid_npz_filepath = '../../data/data_{:03d}/valid_dataset.npz'.format(<dataset_id>)
    test_npz_filepath = '../../data/data_{:03d}/test_dataset.npz'.format(<dataset_id>)
    
    
    
    
    
    
