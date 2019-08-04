from autodisc.representations.static.pytorchnnrepresentation import helper
import torch
from torch import nn


class BetaVAEClassifier(nn.Module):
    '''
    BetaVAE with Classifier network architecture
    '''
    def __init__(self, n_channels = 1, n_latents = 2, input_size = (100,100), beta = 1.0, gamma = 1.0, num_classes = 2, use_gpu = True, add_var_to_KLD_loss = True, **kwargs):
        super(BetaVAEClassifier, self).__init__()
        self.init_params = locals()
        
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        
        # network parameters
        self.n_channels = n_channels
        self.n_latents = n_latents
        self.input_size = input_size
        self.beta = beta
        self.num_classes = num_classes
        self.gamma = gamma
        self.add_var_to_KLD_loss = add_var_to_KLD_loss
        
        # network architecture
        self.encoder = nn.Sequential(
                    nn.Conv2d( self.n_channels, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    helper.Flatten(),
                    nn.Linear(32 * helper.conv2d_output_flatten_size(self.input_size, n_conv=4, kernels_size=[4,4,4,4], strides=[2,2,2,2], pads=[1,1,1,1], dils=[1,1,1,1]), 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2 * self.n_latents)
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n_latents, 256),
                nn.ReLU(),
                nn.Linear(256, 16 * 16 * 32),
                nn.ReLU(),
                helper.Channelize(32, 16, 16),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  #The padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=1) 
                )
        
        self.fc = nn.Sequential(
                nn.Linear(self.n_latents, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
                )

    def encode(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        return torch.chunk(self.encoder(x), 2, dim=1)
    
    def decode(self, z):
        if self.use_gpu and not z.is_cuda:
           z = z.cuda()
        return self.decoder(z)
    
    def classify(self, z):
        if self.use_gpu and not z.is_cuda:
           z = z.cuda()
        return self.fc(z)

    def forward(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {'recon_x': self.decode(z), 'recon_y': self.classify(z), 'mu': mu, 'logvar': logvar}
    
    def reparameterize(self, mu, logvar):
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def calc(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        mu, logvar = self.encode(x)
        return mu
    
    def train_loss(self, outputs, inputs):
        """ train loss:
        recon_x: reconstructed images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        x = inputs['image']
        y = inputs['label']
        recon_x = outputs['recon_x']
        recon_y = outputs['recon_y']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not y.is_cuda:
            y = y.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not recon_y.is_cuda:
            recon_y = recon_y.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        BCE = helper.BCE_with_digits_loss(recon_x, x)  
        KLD, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        CE = helper.CE_loss(recon_y, y)
        total = (BCE + self.beta * (KLD + float(self.add_var_to_KLD_loss) * KLD_var) + self.gamma * CE) 
       
        return {'total': total, 'BCE': BCE, 'KLD': KLD, 'KLD_var': KLD_var, 'CE': CE}
    
    def valid_losses(self, outputs, inputs):
        """ train loss:
        recon_x: reconstructed images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        x = inputs['image']
        y = inputs['label']
        recon_x = outputs['recon_x']
        recon_y = outputs['recon_y']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not y.is_cuda:
            y = y.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not recon_y.is_cuda:
            recon_y = recon_y.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        BCE = helper.BCE_with_digits_loss(recon_x, x)  
        KLD, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        CE = helper.CE_loss(recon_y, y)
        total = (BCE + self.beta * (KLD + float(self.add_var_to_KLD_loss) * KLD_var) + self.gamma * CE) 
        
        return {'total': total, 'BCE': BCE, 'KLD': KLD, 'KLD_var': KLD_var, 'CE': CE}
    

    
class BetaVAE(nn.Module):
    '''
    BetaVAE network architecture
    '''
    def __init__(self, n_channels = 1, n_latents = 2, input_size = (100,100), beta = 1.0, use_gpu = True, add_var_to_KLD_loss = True, **kwargs):
        super(BetaVAE, self).__init__()
        self.init_params = locals()
        
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        
        # network parameters
        self.n_channels = n_channels
        self.n_latents = n_latents
        self.input_size = input_size
        self.beta = beta
        self.add_var_to_KLD_loss = add_var_to_KLD_loss
        
        # network architecture
        self.encoder = nn.Sequential(
                    nn.Conv2d( self.n_channels, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    #helper.LinearFromFlatten(256),
                    helper.Flatten(),
                    nn.Linear(32 * helper.conv2d_output_flatten_size(self.input_size, n_conv=4, kernels_size=[4,4,4,4], strides=[2,2,2,2], pads=[1,1,1,1], dils=[1,1,1,1]), 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2 * self.n_latents)
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n_latents, 256),
                nn.ReLU(),
                nn.Linear(256, 16 * 16 * 32),
                nn.ReLU(),
                helper.Channelize(32, 16, 16),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  #The padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=1) 
                )
        '''
        self.encoder = nn.Sequential(
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d( self.n_channels, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.Flatten(),
                    nn.Linear(32 * helper.conv2d_output_flatten_size(self.input_size, n_conv=4, kernels_size=[4,4,4,4], strides=[2,2,2,2], pads=[1,1,1,1], dils=[1,1,1,1]), 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2 * self.n_latents)
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n_latents, 256),
                nn.ReLU(),
                nn.Linear(256, 16 * 16 * 32),
                nn.ReLU(),
                helper.Channelize(32, 16, 16),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=4-1),  #The padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
                nn.ReLU(),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=4-1),  
                nn.ReLU(),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=4-1),  
                nn.ReLU(),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=4-1) 
                )
        '''

    def encode(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        return torch.chunk(self.encoder(x), 2, dim=1)
    
    def decode(self, z):
        if self.use_gpu and not z.is_cuda:
           z = z.cuda()
        return self.decoder(z)

    def forward(self, x):
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {'recon_x': self.decode(z), 'mu': mu, 'logvar': logvar}
    
    def reparameterize(self, mu, logvar):
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def calc(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
        mu, logvar = self.encode(x)
        return mu
    
    def train_loss(self, outputs, inputs):
        """ train loss:
        recon_x: reconstructed images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        BCE = helper.BCE_with_digits_loss(recon_x, x)  
        KLD, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        total = (BCE + self.beta * (KLD + float(self.add_var_to_KLD_loss) * KLD_var)) 
       
        return {'total': total, 'BCE': BCE, 'KLD': KLD, 'KLD_var': KLD_var}
    
    def valid_losses(self, outputs, inputs):
        """ train loss:
        recon_x: reconstructed images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        x = inputs['image']
        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        if self.use_gpu and not x.is_cuda:
            x = x.cuda()
            
        if self.use_gpu and not recon_x.is_cuda:
            recon_x = recon_x.cuda()
            
        if self.use_gpu and not mu.is_cuda:
            mu = mu.cuda()
            
        if self.use_gpu and not logvar.is_cuda:
            logvar = logvar.cuda()
        
        BCE = helper.BCE_with_digits_loss(recon_x, x)  
        KLD, KLD_per_latent_dim, KLD_var = helper.KLD_loss(mu, logvar)
        total = (BCE + self.beta * (KLD + float(self.add_var_to_KLD_loss) * KLD_var)) 
       
        return {'total': total, 'BCE': BCE, 'KLD': KLD, 'KLD_var': KLD_var}
    
    
    
    
    
class AE(nn.Module):
    '''
    AE network architecture
    '''
    def __init__(self, n_channels = 1, n_latents = 2, input_size = (100,100), use_gpu = True, **kwargs):
        super(AE, self).__init__()
        self.init_params = locals()
        
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        
        # network parameters
        self.n_channels = 1
        self.n_latents = n_latents
        self.input_size = input_size
        
        # network architecture
        self.encoder = nn.Sequential(
                    nn.Conv2d( self.n_channels, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    helper.LinearFromFlatten(256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.n_latents)
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n_latents, 256),
                nn.ReLU(),
                nn.Linear(256, 16 * 16 * 32),
                nn.ReLU(),
                helper.Channelize(32, 16, 16),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  #The padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=1) 
                )
        '''
        self.encoder = nn.Sequential(
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d( self.n_channels, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.SphericPad (padding_size=1),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    helper.Flatten(),
                    nn.Linear(32 * helper.conv2d_output_flatten_size(self.input_size, n_conv=4, kernels_size=[4,4,4,4], strides=[2,2,2,2], pads=[1,1,1,1], dils=[1,1,1,1]), 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.n_latents)
            )

        self.decoder = nn.Sequential(
                nn.Linear(self.n_latents, 256),
                nn.ReLU(),
                nn.Linear(256, 16 * 16 * 32),
                nn.ReLU(),
                helper.Channelize(32, 16, 16),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=4-1),  #The padding argument effectively adds kernel_size - 1 - padding amount of zero padding to both sizes of the input
                nn.ReLU(),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=4-1),  
                nn.ReLU(),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=4-1),  
                nn.ReLU(),
                helper.SphericPad (padding_size=1),
                nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=4-1) 
                )
        '''



    def forward(self, x):
        if self.use_gpu:
            x = x.cuda()
        z = self.encoder(x)
        recon_x = self.decoder(z)

        return {'recon_x': recon_x, 'mu': z, 'logvar': torch.zeros_like(z)}
        
    def calc(self, x):
        ''' the function calc outputs a representation vector of size batch_size*n_latents'''
        if self.use_gpu:
            x = x.cuda()
        z = self.encoder(x)
        return z
    
    def train_loss(self, output_x, x):
        """ train loss """
        recon_x = output_x ['recon_x']
        return helper.MSE_loss(recon_x, x)  
    
    def valid_losses(self, output_x, x):
        total = self.train_loss (output_x, x)
        return {'total': total}
