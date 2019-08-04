import os
import json
import re
import numpy as np                          # pip3 install numpy
from scipy import ndimage
import reikna.fft, reikna.cluda             # pip3 install pyopencl/pycuda, reikna
import warnings
import autodisc as ad
from collections import OrderedDict
import fractions
import collections
import torch
from autodisc.representations.static.pytorchnnrepresentation.helper import SphericPad

warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # suppress warning from snd.zoom()

ROUND = 10

EPS = 0.0001


def rle2arr(st):
    '''
    Transforms an RLE string to a numpy array.

    Code from Bert Chan.

    :param st Description of the array in RLE format.
    :return Numpy array.
    '''

    rle_groups = re.findall("(\d*)([p-y]?[.boA-X$])", st.rstrip('!'))  # [(2 yO)(1 $)(1 yO)]
    code_list = sum([[c] * (1 if n == '' else int(n)) for n, c in rle_groups], [])  # [yO yO $ yO]
    code_arr = [l.split(',') for l in ','.join(code_list).split('$')]  # [[yO yO] [yO]]
    V = [[0 if c in ['.', 'b'] else 255 if c == 'o' else ord(c) - ord('A') + 1 if len(c) == 1 else (ord(c[0]) - ord(
        'p')) * 24 + (ord(c[1]) - ord('A') + 25) for c in row if c != ''] for row in code_arr]  # [[255 255] [255]]
    maxlen = len(max(V, key=len))
    A = np.array([row + [0] * (maxlen - len(row)) for row in V]) / 255  # [[1 1] [1 0]]
    return A


class Board:

    def __init__(self, size=(10,10)):
        self.params = {'R':10, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1}
        self.cells = np.zeros(size)


    def clear(self):
        self.cells.fill(0)


class Automaton:

    kernel_core = {
        0: lambda r: (4 * r * (1-r))**4,  # polynomial (quad4)
        1: lambda r: np.exp( 4 - 1 / (r * (1-r)) ),  # exponential / gaussian bump (bump4)
        2: lambda r, q=1/4: (r>=q)*(r<=1-q),  # step (stpz1/4)
        3: lambda r, q=1/4: (r>=q)*(r<=1-q) + (r<q)*0.5 # staircase (life)
    }
    field_func = {
        0: lambda n, m, s: np.maximum(0, 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,  # polynomial (quad4)
        1: lambda n, m, s: np.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1,  # exponential / gaussian (gaus)
        2: lambda n, m, s: (np.abs(n-m)<=s) * 2 - 1  # step (stpz)
    }


    def __init__(self, world):
        self.world = world
        self.world_FFT = np.zeros(world.cells.shape)
        self.potential_FFT = np.zeros(world.cells.shape)
        self.potential = np.zeros(world.cells.shape)
        self.field = np.zeros(world.cells.shape)
        self.field_old = None
        #self.change = np.zeros(world.cells.shape)
        self.X = None
        self.Y = None
        self.D = None
        self.gen = 0
        self.time = 0
        self.is_multi_step = False
        self.is_soft_clip = False
        self.is_inverted = False
        self.kn = 1
        self.gn = 1
        self.is_gpu = True
        self.has_gpu = True
        self.compile_gpu(self.world.cells)
        self.calc_kernel()


    def kernel_shell(self, r):
        k = len(self.world.params['b'])
        kr = k * r
        bs = np.array([float(f) for f in self.world.params['b']])
        b = bs[np.minimum(np.floor(kr).astype(int), k-1)]
        kfunc = Automaton.kernel_core[(self.world.params.get('kn') or self.kn) - 1]
        return (r<1) * kfunc(np.minimum(kr % 1, 1)) * b


    @staticmethod
    def soft_max(x, m, k):
        ''' Soft maximum: https://www.johndcook.com/blog/2010/01/13/soft-maximum/ '''
        return np.log(np.exp(k*x) + np.exp(k*m)) / k


    @staticmethod
    def soft_clip(x, min, max, k):
        a = np.exp(k*x)
        b = np.exp(k*min)
        c = np.exp(-k*max)
        return np.log( 1/(a+b)+c ) / -k


    def compile_gpu(self, A):
        ''' Reikna: http://reikna.publicfields.net/en/latest/api/computations.html '''
        self.gpu_api = self.gpu_thr = self.gpu_fft = self.gpu_fftshift = None
        try:
            self.gpu_api = reikna.cluda.any_api()
            self.gpu_thr = self.gpu_api.Thread.create()
            self.gpu_fft = reikna.fft.FFT(A.astype(np.complex64)).compile(self.gpu_thr)
            self.gpu_fftshift = reikna.fft.FFTShift(A.astype(np.float32)).compile(self.gpu_thr)
        except Exception as exc:
            # if str(exc) == "No supported GPGPU APIs found":
            self.has_gpu = False
            self.is_gpu = False
            # print(exc)
            # raise exc


    def run_gpu(self, A, cpu_func, gpu_func, dtype, **kwargs):
        if self.is_gpu and self.gpu_thr and gpu_func:
            op_dev = self.gpu_thr.to_device(A.astype(dtype))
            gpu_func(op_dev, op_dev, **kwargs)
            return op_dev.get()
        else:
            return cpu_func(A)

    def fft(self, A): return self.run_gpu(A, np.fft.fft2, self.gpu_fft, np.complex64)
    def ifft(self, A): return self.run_gpu(A, np.fft.ifft2, self.gpu_fft, np.complex64, inverse=True)
    def fftshift(self, A): return self.run_gpu(A, np.fft.fftshift, self.gpu_fftshift, np.float32)


    def calc_once(self):
        A = self.world.cells
        self.world_FFT = self.fft(A)
        self.potential_FFT = self.kernel_FFT * self.world_FFT
        self.potential = self.fftshift(np.real(self.ifft(self.potential_FFT)))
        gfunc = Automaton.field_func[(self.world.params.get('gn') or self.gn) - 1]
        self.field = gfunc(self.potential, self.world.params['m'], self.world.params['s'])
        dt = 1 / self.world.params['T']
        if self.is_multi_step and self.field_old:
            D = 1/2 * (3 * self.field - self.field_old)
            self.field_old = self.field.copy()
        else:
            D = self.field
        if not self.is_soft_clip:
            A_new = np.clip(A + dt * D, 0, 1)  # A_new = A + dt * np.clip(D, -A/dt, (1-A)/dt)
        else:
            A_new = Automaton.soft_clip(A + dt * D, 0, 1, 1/dt)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
        #self.change = (A_new - A) / dt
        self.world.cells = A_new
        #self.gen += 1
        #self.time = round(self.time + dt, ROUND)
        if self.is_gpu:
            self.gpu_thr.synchronize()


    def calc_kernel(self):

        size_y = self.world.cells.shape[0]
        size_x = self.world.cells.shape[1]

        I, J = np.meshgrid(np.arange(size_x), np.arange(size_y))
        self.X = (I - int(size_x / 2)) / self.world.params['R']
        self.Y = (J - int(size_y / 2)) / self.world.params['R']

        self.D = np.sqrt(self.X**2 + self.Y**2)

        self.kernel = self.kernel_shell(self.D)
        self.kernel_sum = np.sum(self.kernel)
        self.kernel_norm = self.kernel / self.kernel_sum
        self.kernel_FFT = self.fft(self.kernel_norm)
        self.kernel_updated = False
        


    def reset(self):
        #self.gen = 0
        #self.time = 0
        self.field_old = None
        
        
        
'''---------------------------------------------------------------
    AUTOMATON PYTORCH VERSION
-------------------------------------------------------------------'''
def complex_mult_torch(X, Y):
    """ Computes the complex multiplication in Pytorch when the tensor last dimension is 2: 0 is the real component and 1 the imaginary one"""
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)
        
def roll_n(X, axis, n):
    """ Rolls a tensor with a shift n on the specified axis"""
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

    
class LeniaStepFFT(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the fft version"""
    
    def __init__(self, R, b, kn, gn, m, s, T, is_soft_clip, is_gpu, size_y, size_x):
        super(LeniaStepFFT, self).__init__()
        self.R = R
        self.T = T
        self.dt = float (1.0 / T)
        self.b = b
        self.kn = kn
        self.gn = gn
        self.m = m
        self.s = s
        self.spheric_pad = SphericPad(int(self.R))
        self.is_soft_clip = is_soft_clip
        self.is_gpu = is_gpu
        self.size_y = size_y
        self.size_x = size_x
        
        self.compute_kernel()
        
    
    def compute_kernel(self):
        size_y = self.size_y
        size_x = self.size_x

        # implementation of meshgrid in torch
        x = torch.arange(size_x)
        y = torch.arange(size_y)
        xx = x.repeat(size_y, 1)
        yy = y.view(-1,1).repeat(1, size_x)
        X = (xx - int(size_x / 2)).float() / float(self.R)
        Y =  (yy - int(size_y / 2)).float() / float(self.R)
        
        # distance to center in normalized space
        D = torch.sqrt(X**2 + Y**2)

        # kernel
        k = len(self.b)
        kr = k * D
        bs = torch.tensor([float(f) for f in self.b])
        b = bs[torch.min(torch.floor(kr).long(), (k-1)*torch.ones_like(kr).long())]
        kfunc = AutomatonPytorch.kernel_core[self.kn - 1]
        kernel = (D<1).float() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
        # fft of the kernel
        self.kernel_FFT = torch.rfft(self.kernel_norm, signal_ndim=2, onesided=False)
        
        self.kernel_updated = False
    
    
    def forward(self, input):
        if self.is_gpu:
            input = input.cuda()
            self.kernel_FFT = self.kernel_FFT.cuda()

        self.world_FFT = torch.rfft(input, signal_ndim=2, onesided=False)
        self.potential_FFT = complex_mult_torch(self.kernel_FFT, self.world_FFT)
        self.potential = torch.irfft(self.potential_FFT, signal_ndim=2, onesided=False)
        self.potential = roll_n(self.potential, 3, self.potential.size(3)//2)
        self.potential = roll_n(self.potential, 2, self.potential.size(2)//2)

        gfunc = AutomatonPytorch.field_func[self.gn]
        self.field = gfunc(self.potential, self.m, self.s)
        
        if not self.is_soft_clip:
            output_img = torch.clamp(input + self.dt * self.field, min=0., max=1.)
        else:
            output_img = AutomatonPytorch.soft_clip(input + self.dt * self.field, 0, 1, self.T)
        
        return output_img

   
    
class LeniaStepConv2d(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the conv2d version"""
    
    def __init__(self, R, b, kn, gn, m, s, T, is_soft_clip, is_gpu):
        super(LeniaStepConv2d, self).__init__()
        self.R = R
        self.T = T
        self.dt =  float (1.0 / T)
        self.b = b
        self.kn = kn
        self.gn = gn
        self.m = m
        self.s = s
        self.spheric_pad = SphericPad(int(self.R))
        self.is_soft_clip = is_soft_clip
        self.is_gpu = is_gpu
        
        self.compute_kernel()

    
    def compute_kernel(self):
        size_y = 2 * self.R + 1
        size_x = 2 * self.R + 1

        # implementation of meshgrid in torch
        x = torch.arange(size_x)
        y = torch.arange(size_y)
        xx = x.repeat(size_y, 1)
        yy = y.view(-1,1).repeat(1, size_x)
        X = (xx - int(size_x / 2)).float() / float(self.R)
        Y =  (yy - int(size_y / 2)).float() / float(self.R)
        
        # distance to center in normalized space
        D = torch.sqrt(X**2 + Y**2)
        
        # kernel
        k = len(self.b)
        kr = k * D
        bs = torch.tensor([float(f) for f in self.b])
        b = bs[torch.min(torch.floor(kr).long(), (k-1)*torch.ones_like(kr).long())]
        kfunc = AutomatonPytorch.kernel_core[self.kn - 1]
        kernel = (D<1).float() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
        
        self.kernel_updated = False
    
    
    def forward(self, input):
        if self.is_gpu:
            input = input.cuda()
            self.kernel_norm = self.kernel_norm.cuda()
            
        self.potential = torch.nn.functional.conv2d(self.spheric_pad(input), weight = self.kernel_norm)
        gfunc = AutomatonPytorch.field_func[self.gn]
        self.field = gfunc(self.potential, self.m, self.s)
        
        if not self.is_soft_clip:
            output_img = torch.clamp(input + self.dt * self.field, 0, 1)  # A_new = A + dt * torch.clamp(D, -A/dt, (1-A)/dt)
        else:
            output_img = AutomatonPytorch.soft_clip(input + self.dt * self.field, 0, 1, self.T)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
        
        return output_img



class AutomatonPytorch:
    kernel_core = {
        0: lambda r: (4 * r * (1-r))**4,  # polynomial (quad4)
        1: lambda r: torch.exp( 4 - 1 / (r * (1-r)) ),  # exponential / gaussian bump (bump4)
        2: lambda r, q=1/4: (r>=q)*(r<=1-q),  # step (stpz1/4)
        3: lambda r, q=1/4: (r>=q)*(r<=1-q) + (r<q)*0.5 # staircase (life)
    }
    field_func = {
        0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,  # polynomial (quad4)
        1: lambda n, m, s: torch.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1,  # exponential / gaussian (gaus)
        2: lambda n, m, s: (torch.abs(n-m)<=s) * 2 - 1  # step (stpz)
    }
    
    @staticmethod
    def soft_max(x, m, k):
        return torch.log(torch.exp(k*x) + torch.exp(k*m)) / k

    @staticmethod
    def soft_clip(x, min, max, k):
        a = torch.exp(k*x)
        b = torch.exp(k*min)
        c = torch.exp(-k*max)
        return torch.log( 1/(a+b)+c ) / -k


    def __init__(self, world, version = 'fft'):
        self.world = world
        #self.world_FFT = np.zeros(world.cells.shape)
        #self.potential_FFT = np.zeros(world.cells.shape)
        #self.potential = np.zeros(world.cells.shape)
        #self.field = np.zeros(world.cells.shape)
        #self.field_old = None
        #self.change = np.zeros(world.cells.shape)
        self.X = None
        self.Y = None
        self.D = None
        #self.gen = 0
        #self.time = 0
        self.is_multi_step = False
        self.is_soft_clip = False
        self.is_inverted = False
        self.kn = 1
        self.gn = 1
        # look if gpu is available
        if torch.cuda.is_available():
            is_gpu = True
        else:
            is_gpu = False
        # initialization of the pytorch model to perform one step in Lenia
        if version == 'fft':
            self.model = LeniaStepFFT(self.world.params['R'], self.world.params['b'], (self.world.params.get('kn') or self.kn), (self.world.params.get('gn') or self.gn), self.world.params['m'], self.world.params['s'], self.world.params['T'], self.is_soft_clip, is_gpu, self.world.cells.shape[0], self.world.cells.shape[1])
        elif version == 'conv2d':
            self.model = LeniaStepConv2d(self.world.params['R'], self.world.params['b'], (self.world.params.get('kn') or self.kn), (self.world.params.get('gn') or self.gn), self.world.params['m'], self.world.params['s'], self.world.params['T'], self.is_soft_clip, is_gpu)
        else:
            raise ValueError('Lenia pytorch automaton step calculation can be done with fft or conv 2d')
        if is_gpu:
            self.model = self.model.cuda()    
    
    def calc_once(self):
        A = torch.from_numpy(self.world.cells).unsqueeze(0).unsqueeze(0).float()
        A_new = self.model(A)
        #A = A[0,0,:,:].cpu().numpy()
        A_new = A_new[0,0,:,:].cpu().numpy()
        #self.change = (A_new - A) / (1/self.world.params['T'])
        self.world.cells = A_new
        #self.gen += 1
        #self.time = round(self.time + (1/self.world.params['T']), ROUND)

    def reset(self):
        pass
        #self.gen = 0
        #self.time = 0
        #self.field_old = None
 
     
class Lenia(ad.core.System):

    @staticmethod
    def default_config():
        def_config = ad.core.System.default_config()
        def_config.version = 'pytorch_fft' # reikna_fft, pytorch_fft, pytorch_conv2d
        return def_config


    @staticmethod
    def default_system_parameters():
        def_params = ad.core.System.default_system_parameters()
        def_params.size_y = 100
        def_params.size_x = 100
        def_params.R = 13
        def_params.T = 10
        def_params.b = [1]
        def_params.m = 0.15
        def_params.s = 0.017
        def_params.kn = 1
        def_params.gn = 1
        def_params.init_state = np.zeros((def_params.size_y, def_params.size_x))
        return def_params


    def default_statistics(self):
        def_stats = super().default_statistics()
        def_stats.append(LeniaStatistics(self))

        return def_stats


    def __init__(self, statistics=None, system_parameters=None, config=None, **kwargs):

        super().__init__(statistics=statistics, system_parameters=system_parameters, config=config, **kwargs)

        self.run_parameters = None
        self.world = None
        self.automaton = None


    def init_run(self, run_parameters=None):

        if run_parameters is None:
            self.run_parameters = self.system_parameters
        else:
            self.run_parameters = {**self.system_parameters, **run_parameters}

        self.world = Board((self.run_parameters['size_y'], self.run_parameters['size_x']))
        self.world.cells = self.run_parameters["init_state"]
        self.world.params = self.run_parameters

        if np.min(self.world.cells) < 0 or np.max(self.world.cells) > 1:
            raise Warning('The given initial state has values below 0 and\or above 1. It will be clipped to the range [0, 1]')
            self.world.cells = np.clip(self.world.cells, 0, 1)

        if self.config.version.lower() == 'pytorch_fft':
            self.automaton = AutomatonPytorch(self.world, version='fft')
        elif self.config.version.lower() == 'reikna_fft':
            self.automaton = Automaton(self.world)
        elif self.config.version.lower() == 'pytorch_conv2d':
            self.automaton = AutomatonPytorch(self.world, version='conv2d')
        else:
            raise ValueError('Unknown lenia version (config.version = {!r})'.format(self.config.version))

        return self.world.cells


    def step(self, step_idx):
        self.automaton.calc_once()

        # for some invalid parameters become the cells nan
        # this makes problems with the computation of the statistics
        # therefore, assume that all nan cells are 0
        self.world.cells[np.isnan(self.world.cells)] = 0

        return self.world.cells


    def stop(self):
        pass



class LeniaStatistics(ad.core.SystemStatistic):
    '''Default statistics for the lenia system.'''

    DISTANCE_WEIGHT = 2  # 1=linear, 2=quadratic, ...

    @staticmethod
    def calc_statistic_diff(statistic_names, stat1, stat2, nan_value_diff=1.0, nan_nan_diff=0.0):

        if isinstance(stat1, list) or isinstance(stat2, list):
            raise NotImplementedError('Difference between statistics given as lists are not implemented!')

        if not isinstance(statistic_names, list):
            statistic_names = [statistic_names]
            stat1 = [stat1]
            stat2 = [stat2]

        # assume default difference for all
        diff = stat1 - stat2

        # check if there are angle statistics and calculate there difference appropriately
        statistic_names_ndarray = np.array(statistic_names)
        angle_statistics_inds = (statistic_names_ndarray == 'activation_center_movement_angle') \
                                | (statistic_names_ndarray == 'activation_center_movement_angle_mean') \
                                | (statistic_names_ndarray == 'positive_growth_center_movement_angle') \
                                | (statistic_names_ndarray == 'positive_growth_center_movement_angle_mean')

        for angle_stat_idx in np.where(angle_statistics_inds)[0]:
            diff[angle_stat_idx] = ad.helper.misc.angle_difference_degree(stat1[angle_stat_idx], stat2[angle_stat_idx])

        # if both statistics are nan, then the difference is nan_nan_diff (default=0)
        diff[np.isnan(stat1) & np.isnan(stat2)] = nan_nan_diff

        # if one statistic is nan, then the current diff is nan, then use nan_value_diff
        diff[np.isnan(diff)] = nan_value_diff

        return diff


    @staticmethod
    def calc_goalspace_distance(points1, points2, config):

        diff = LeniaStatistics.calc_statistic_diff(config.statistics, points1, points2)

        if len(diff) == 0:
            dist = np.array([])
        elif np.ndim(diff) == 1:
            dist = np.linalg.norm(diff)
        else:
            dist = np.linalg.norm(diff, axis=1)

        return dist


    def __init__(self, system):

        super().__init__(system)

        # statistics

        self.data['is_dead'] = []

        self.data['activation_mass'] = []
        self.data['activation_mass_mean'] = []
        self.data['activation_mass_std'] = []

        self.data['activation_volume'] = []
        self.data['activation_volume_mean'] = []
        self.data['activation_volume_std'] = []

        self.data['activation_density'] = []
        self.data['activation_density_mean'] = []
        self.data['activation_density_std'] = []

        self.data['activation_center_position'] = []

        self.data['activation_center_velocity'] = []
        self.data['activation_center_velocity_mean'] = []
        self.data['activation_center_velocity_std'] = []

        self.data['activation_center_movement_angle'] = []
        self.data['activation_center_movement_angle_mean'] = []
        self.data['activation_center_movement_angle_std'] = []

        self.data['activation_center_movement_angle_velocity'] = []
        self.data['activation_center_movement_angle_velocity_mean'] = []
        self.data['activation_center_movement_angle_velocity_std'] = []

        self.data['activation_mass_asymmetry'] = []
        self.data['activation_mass_asymmetry_mean'] = []
        self.data['activation_mass_asymmetry_std'] = []

        self.data['activation_mass_distribution'] = []
        self.data['activation_mass_distribution_mean'] = []
        self.data['activation_mass_distribution_std'] = []

        self.data['activation_hu1'] = []
        self.data['activation_hu1_mean'] = []
        self.data['activation_hu1_std'] = []

        self.data['activation_hu2'] = []
        self.data['activation_hu2_mean'] = []
        self.data['activation_hu2_std'] = []

        self.data['activation_hu3'] = []
        self.data['activation_hu3_mean'] = []
        self.data['activation_hu3_std'] = []

        self.data['activation_hu4'] = []
        self.data['activation_hu4_mean'] = []
        self.data['activation_hu4_std'] = []

        self.data['activation_hu5'] = []
        self.data['activation_hu5_mean'] = []
        self.data['activation_hu5_std'] = []

        self.data['activation_hu6'] = []
        self.data['activation_hu6_mean'] = []
        self.data['activation_hu6_std'] = []

        self.data['activation_hu7'] = []
        self.data['activation_hu7_mean'] = []
        self.data['activation_hu7_std'] = []

        self.data['activation_hu8'] = []
        self.data['activation_hu8_mean'] = []
        self.data['activation_hu8_std'] = []

        self.data['activation_flusser9'] = []
        self.data['activation_flusser9_mean'] = []
        self.data['activation_flusser9_std'] = []

        self.data['activation_flusser10'] = []
        self.data['activation_flusser10_mean'] = []
        self.data['activation_flusser10_std'] = []

        self.data['activation_flusser11'] = []
        self.data['activation_flusser11_mean'] = []
        self.data['activation_flusser11_std'] = []

        self.data['activation_flusser12'] = []
        self.data['activation_flusser12_mean'] = []
        self.data['activation_flusser12_std'] = []

        self.data['activation_flusser13'] = []
        self.data['activation_flusser13_mean'] = []
        self.data['activation_flusser13_std'] = []

        self.data['positive_growth_mass'] = []
        self.data['positive_growth_mass_mean'] = []
        self.data['positive_growth_mass_std'] = []

        self.data['positive_growth_volume'] = []
        self.data['positive_growth_volume_mean'] = []
        self.data['positive_growth_volume_std'] = []

        self.data['positive_growth_density'] = []
        self.data['positive_growth_density_mean'] = []
        self.data['positive_growth_density_std'] = []

        self.data['positive_growth_center_position'] = []

        self.data['positive_growth_center_velocity'] = []
        self.data['positive_growth_center_velocity_mean'] = []
        self.data['positive_growth_center_velocity_std'] = []

        self.data['positive_growth_center_movement_angle'] = []
        self.data['positive_growth_center_movement_angle_mean'] = []
        self.data['positive_growth_center_movement_angle_std'] = []

        self.data['positive_growth_center_movement_angle_velocity'] = []
        self.data['positive_growth_center_movement_angle_velocity_mean'] = []
        self.data['positive_growth_center_movement_angle_velocity_std'] = []

        self.data['activation_positive_growth_centroid_distance'] = []
        self.data['activation_positive_growth_centroid_distance_mean'] = []
        self.data['activation_positive_growth_centroid_distance_std'] = []

        # other
        self.distance_weight_matrix = LeniaStatistics.calc_distance_matrix(system.system_parameters.size_y,
                                                                           system.system_parameters.size_x)

        self.angles_from_middle = None


    def reset(self):
        # set all statistics to zero
        self.data = dict.fromkeys(self.data, [])


    def calc_after_run(self, system, all_obs):
        '''Calculates the final statistics for lenia observations after a run is completed'''

        self.reset()

        num_of_obs = len(all_obs)

        activation_mass_data = np.ones(num_of_obs) * np.nan
        activation_volume_data = np.ones(num_of_obs) * np.nan
        activation_density_data = np.ones(num_of_obs) * np.nan

        activation_center_position_data = np.ones((num_of_obs, 2)) * np.nan
        activation_center_velocity_data = np.ones(num_of_obs) * np.nan
        activation_center_movement_angle_data = np.ones(num_of_obs) * np.nan
        activation_center_movement_angle_velocity_data = np.ones(num_of_obs) * np.nan

        activation_mass_asymmetry_data = np.ones(num_of_obs) * np.nan
        activation_mass_distribution_data = np.ones(num_of_obs) * np.nan

        activation_hu1_data = np.ones(num_of_obs) * np.nan
        activation_hu2_data = np.ones(num_of_obs) * np.nan
        activation_hu3_data = np.ones(num_of_obs) * np.nan
        activation_hu4_data = np.ones(num_of_obs) * np.nan
        activation_hu5_data = np.ones(num_of_obs) * np.nan
        activation_hu6_data = np.ones(num_of_obs) * np.nan
        activation_hu7_data = np.ones(num_of_obs) * np.nan
        activation_hu8_data = np.ones(num_of_obs) * np.nan
        activation_flusser9_data = np.ones(num_of_obs) * np.nan
        activation_flusser10_data = np.ones(num_of_obs) * np.nan
        activation_flusser11_data = np.ones(num_of_obs) * np.nan
        activation_flusser12_data = np.ones(num_of_obs) * np.nan
        activation_flusser13_data = np.ones(num_of_obs) * np.nan

        positive_growth_mass_data = np.ones(num_of_obs) * np.nan
        positive_growth_volume_data = np.ones(num_of_obs) * np.nan
        positive_growth_density_data = np.ones(num_of_obs) * np.nan

        positive_growth_center_position_data = np.ones((num_of_obs, 2)) * np.nan
        positive_growth_center_velocity_data = np.ones(num_of_obs) * np.nan
        positive_growth_center_movement_angle_data = np.ones(num_of_obs) * np.nan
        positive_growth_center_movement_angle_velocity_data = np.ones(num_of_obs) * np.nan

        activation_positive_growth_centroid_distance_data = np.ones(num_of_obs) * np.nan


        # positive_growth_data = np.ones(num_of_obs) * np.nan
        # positive_growth_volume_data = np.ones(num_of_obs) * np.nan
        # positive_growth_density_data = np.ones(num_of_obs) * np.nan


        size_y = all_obs[0].shape[0]
        size_x = all_obs[0].shape[1]
        num_of_cells = size_y * size_x

        # calc initial center of mass and use it as a reference point to "center" the world around it
        # in consequetive steps, recalculate the center of mass and "recenter" the wolrd around them
        #mid_y = int((size_y-1) / 2)
        #mid_x = int((size_x-1) / 2)
        mid_y = (size_y - 1) / 2
        mid_x = (size_x - 1) / 2
        mid = np.array([mid_y, mid_x])

        # prepare the angles of the vectors from the middle point for each point in the env, used to compute the mass asymmetry
        # only recompute for first calculation of statistics (self.angles_from_middle is None) or if the observation size changed
        if self.angles_from_middle is None or self.angles_from_middle.shape[0] != size_y or self.angles_from_middle.shape[1] != size_x:
            self.angles_from_middle = np.ones((size_y,size_x))*np.nan
            for y in range(size_y):
                for x in range(size_x):
                    vec = [mid_y-y, x-mid_x]
                    self.angles_from_middle[y][x] = ad.helper.misc.angle_of_vec_degree([vec[1], vec[0]])

        activation_center_of_mass = np.array(LeniaStatistics.center_of_mass(all_obs[0]))
        activation_shift_to_center = mid - activation_center_of_mass

        init_growth = all_obs[1] - all_obs[0]
        positive_growth_center_of_mass = np.array(LeniaStatistics.center_of_mass(init_growth))
        positive_growth_shift_to_center = mid - positive_growth_center_of_mass

        prev_activation_center_movement_angle = np.nan
        prev_positive_growth_center_movement_angle = np.nan

        uncentered_activation_center_position = np.array([np.nan, np.nan])

        for step in range(len(all_obs)):

            activation = all_obs[step]

            # uncentered_activation_center_position = np.array(ndimage.measurements.center_of_mass(activation))
            #
            # # set center to middle if it can not be calculated, for example if all cells are dead
            # if np.isnan(uncentered_activation_center_position[0]) or np.isnan(uncentered_activation_center_position[1]) or \
            #         uncentered_activation_center_position[0] == float('inf') or uncentered_activation_center_position[1] == float('inf'):
            #     uncentered_activation_center_position = mid.copy()

            # shift the system to the last calculated center of mass so that it is in the middle
            # the matrix can only be shifted in discrete values, therefore the shift is transformed to integer
            centered_activation = np.roll(activation, activation_shift_to_center.astype(int), (0, 1))

            # calculate the image moments
            activation_moments = ad.helper.statistics.calc_image_moments(centered_activation)

            # new center of mass
            activation_center_of_mass = np.array([activation_moments.y_avg, activation_moments.x_avg])

            # calculate the change of center as a vector
            activation_shift_from_prev_center = mid - activation_center_of_mass

            # calculate the new shift to center the next obs to the new center
            activation_shift_to_center = activation_shift_to_center + activation_shift_from_prev_center

            # transform the new center, encoded as a shift from the first image, back into the original image coordinates
            uncentered_activation_center_position[0] = (mid_y - activation_shift_to_center[0]) % size_y
            uncentered_activation_center_position[1] = (mid_x - activation_shift_to_center[1]) % size_x
            activation_center_position_data[step] = uncentered_activation_center_position

            # activation mass
            activation_mass = activation_moments.m00
            activation_mass_data[step] = activation_mass / num_of_cells  # activation is number of acitvated cells divided by the number of cells

            # activation volume
            activation_volume = np.sum(activation > EPS)
            activation_volume_data[step] = activation_volume / num_of_cells

            # activation density
            if activation_volume == 0:
                activation_density_data[step] = 0
            else:
                activation_density_data[step] = activation_mass/activation_volume

            # activation moments
            activation_hu1_data[step] = activation_moments.hu1
            activation_hu2_data[step] = activation_moments.hu2
            activation_hu3_data[step] = activation_moments.hu3
            activation_hu4_data[step] = activation_moments.hu4
            activation_hu5_data[step] = activation_moments.hu5
            activation_hu6_data[step] = activation_moments.hu6
            activation_hu7_data[step] = activation_moments.hu7
            activation_hu8_data[step] = activation_moments.hu8
            activation_flusser9_data[step] = activation_moments.flusser9
            activation_flusser10_data[step] = activation_moments.flusser10
            activation_flusser11_data[step] = activation_moments.flusser11
            activation_flusser12_data[step] = activation_moments.flusser12
            activation_flusser13_data[step] = activation_moments.flusser13

            # get velocity and angle of movement
            #   distance between the previous center of mass and the new one is the velocity
            #   angle is computed based on the shift vector
            if step <= 0:
                activation_center_velocity = np.nan
                activation_center_movement_angle = np.nan
                activation_center_movement_angle_velocity = np.nan
                activation_mass_asymmetry = np.nan
            else:
                activation_center_velocity = np.linalg.norm(activation_shift_from_prev_center)

                if activation_center_velocity == 0:
                    activation_center_movement_angle = np.nan
                else:
                    activation_center_movement_angle = ad.helper.misc.angle_of_vec_degree([-1 * activation_shift_from_prev_center[1], activation_shift_from_prev_center[0]])

                # Angular velocity, is the difference between the current and previous angle of movement
                if activation_center_movement_angle is np.nan or prev_activation_center_movement_angle is np.nan:
                    activation_center_movement_angle_velocity = 0
                else:
                    activation_center_movement_angle_velocity = ad.helper.misc.angle_difference_degree(activation_center_movement_angle, prev_activation_center_movement_angle)

                # activation mass asymmetry
                # calculate the angle between the center shift and the angle from the center to each point.
                # if the angle is < 180 the point is on the right side of the movement
                # then use its mass
                activation_right_side_mass = 0

                if np.isnan(activation_center_movement_angle):
                    activation_mass_asymmetry = np.nan
                else:

                    if activation_mass == 0 or activation_mass_asymmetry == num_of_cells:
                        # if all are active or dead then ther is perfect assymetry
                        activation_mass_asymmetry = 0
                    else:

                        # for y in range(size_y):
                        #     for x in range(size_x):
                        #         angle_dist = ad.helper.misc.angle_difference_degree(activation_center_movement_angle, angles_from_middle[y][x])
                        #
                        #         if angle_dist < 180:
                        #             activation_right_side_mass = activation_right_side_mass + activation[y][x]

                        angle_dist = ad.helper.misc.angle_difference_degree(activation_center_movement_angle, self.angles_from_middle)
                        activation_right_side_mass = np.sum(activation[angle_dist < 0])

                        # activation_mass_asymmetry = right_mass - left_mass = right_mass - (mass - right_mass) = 2*right_mass - mass
                        activation_mass_asymmetry = (2 * activation_right_side_mass - activation_mass) / activation_mass

                prev_activation_center_movement_angle = activation_center_movement_angle

            activation_center_velocity_data[step] = activation_center_velocity
            activation_center_movement_angle_data[step] = activation_center_movement_angle
            activation_center_movement_angle_velocity_data[step] = activation_center_movement_angle_velocity
            activation_mass_asymmetry_data[step] = activation_mass_asymmetry

            # mass distribution around the center
            if activation_mass <= EPS:
                activation_mass_distribution = 1.0
            else:
                activation_mass_distribution = np.sum(self.distance_weight_matrix * centered_activation) / np.sum(centered_activation)

            activation_mass_distribution_data[step] = activation_mass_distribution


            ##########################################################################################################################################
            # positive growth statistics

            uncentered_positive_growth_center_position = np.array([np.nan, np.nan])

            if step <= 0:
                positive_growth_mass_data[step] = np.nan
                positive_growth_volume_data[step] = np.nan
                positive_growth_density_data[step] = np.nan

                positive_growth_center_position_data[step] = [np.nan, np.nan]
                positive_growth_center_velocity_data[step] = np.nan
                positive_growth_center_movement_angle_data[step] = np.nan
                positive_growth_center_movement_angle_velocity_data[step] = np.nan
            else:

                positive_growth = np.clip(all_obs[step] - all_obs[step - 1], 0, 1)

                # uncentered_positive_growth_center_position = np.array(StatLenia.center_of_mass(positive_growth))
                #
                # # set center to middle if it can not be calculated, for example if all cells are dead
                # if np.isnan(uncentered_positive_growth_center_position[0]) or np.isnan(uncentered_positive_growth_center_position[1]) or \
                #         uncentered_positive_growth_center_position[0] == float('inf') or uncentered_positive_growth_center_position[1] == float('inf'):
                #     uncentered_positive_growth_center_position = mid.copy()
                #
                # positive_growth_center_position_data[step] = uncentered_positive_growth_center_position

                # shift the system to the last calculated center of mass so that it is in the middle
                # the matrix can only be shifted in discrete values, therefore the shift is transformed to integer
                centered_positive_growth = np.roll(positive_growth, [int(positive_growth_shift_to_center[0]), int(positive_growth_shift_to_center[1])], (0, 1))

                # new center of mass
                positive_growth_center_of_mass = np.array(LeniaStatistics.center_of_mass(centered_positive_growth))

                # calculate the change of center as a vector
                positive_growth_shift_from_prev_center = mid - positive_growth_center_of_mass

                # calculate the new shift to center the next obs to the new center
                positive_growth_shift_to_center = positive_growth_shift_to_center + positive_growth_shift_from_prev_center

                # transform the new center, encoded as a shift from the first image, back into the original image coordinates
                uncentered_positive_growth_center_position[0] = (mid_y - positive_growth_shift_to_center[0]) % size_y
                uncentered_positive_growth_center_position[1] = (mid_x - positive_growth_shift_to_center[1]) % size_x
                positive_growth_center_position_data[step] = uncentered_positive_growth_center_position

                # growth mass
                positive_growth_mass = np.sum(centered_positive_growth)
                positive_growth_mass_data[step] = positive_growth_mass / num_of_cells  # activation is number of acitvated cells divided by the number of cells

                # activation volume
                positive_growth_volume = np.sum( centered_positive_growth > EPS )
                positive_growth_volume_data[step] = positive_growth_volume / num_of_cells

                # activation density
                if positive_growth_volume == 0:
                    positive_growth_density_data[step] = 0
                else:
                    positive_growth_density_data[step] = positive_growth_mass / positive_growth_volume

                # get velocity and angle of movement
                #   distance between the previous center of mass and the new one is the velocity
                #   angle is computed based on the shift vector
                if step <= 1:
                    positive_growth_center_velocity = np.nan
                    positive_growth_center_movement_angle = np.nan
                    positive_growth_center_movement_angle_velocity = np.nan
                else:
                    positive_growth_center_velocity = np.linalg.norm(positive_growth_shift_from_prev_center)

                    if positive_growth_center_velocity == 0:
                        positive_growth_center_movement_angle = np.nan
                    else:
                        positive_growth_center_movement_angle = ad.helper.misc.angle_of_vec_degree([-1 * positive_growth_shift_from_prev_center[1], positive_growth_shift_from_prev_center[0]])

                    # Angular velocity, is the difference between the current and previous angle of movement
                    if positive_growth_center_movement_angle is np.nan or prev_positive_growth_center_movement_angle is np.nan:
                        positive_growth_center_movement_angle_velocity = 0
                    else:
                        positive_growth_center_movement_angle_velocity = ad.helper.misc.angle_difference_degree(positive_growth_center_movement_angle, prev_positive_growth_center_movement_angle)

                    prev_positive_growth_center_movement_angle = positive_growth_center_movement_angle

                positive_growth_center_velocity_data[step] = positive_growth_center_velocity
                positive_growth_center_movement_angle_data[step] = positive_growth_center_movement_angle
                positive_growth_center_movement_angle_velocity_data[step] = positive_growth_center_movement_angle_velocity


            ######################################################################################################################
            # Growth - Activation centroid distance

            if step <= 0:
                activation_positive_growth_centroid_distance_data[step] = np.nan
            else:
                activation_positive_growth_centroid_distance = ad.helper.misc.get_min_distance_on_repeating_2d_array((size_y, size_x), uncentered_activation_center_position, uncentered_positive_growth_center_position)
                activation_positive_growth_centroid_distance_data[step] = activation_positive_growth_centroid_distance


        is_dead = np.all(all_obs[-1] == 1) or np.all(all_obs[-1] == 0)

        self.data['is_dead'] = is_dead

        self.data['activation_mass'] = activation_mass_data
        self.data['activation_mass_mean'] = np.nanmean(activation_mass_data)
        self.data['activation_mass_std'] = np.nanstd(activation_mass_data)

        self.data['activation_volume'] = activation_volume_data
        self.data['activation_volume_mean'] = np.nanmean(activation_volume_data)
        self.data['activation_volume_std'] = np.nanstd(activation_volume_data)

        self.data['activation_density'] = activation_density_data
        self.data['activation_density_mean'] = np.nanmean(activation_density_data)
        self.data['activation_density_std'] = np.nanstd(activation_density_data)

        self.data['activation_center_position'] = activation_center_position_data

        self.data['activation_center_velocity'] = activation_center_velocity_data
        self.data['activation_center_velocity_mean'] = np.nanmean(activation_center_velocity_data)
        self.data['activation_center_velocity_std'] = np.nanstd(activation_center_velocity_data)

        self.data['activation_center_movement_angle'] = activation_center_movement_angle_data
        self.data['activation_center_movement_angle_mean'] = ad.helper.statistics.nan_mean_over_angles_degrees(activation_center_movement_angle_data)
        #self.data['activation_center_movement_angle_std'] = np.nanstd(activation_center_movement_angle_data)

        self.data['activation_center_movement_angle_velocity'] = activation_center_movement_angle_velocity_data
        self.data['activation_center_movement_angle_velocity_mean'] = np.nanmean(activation_center_movement_angle_velocity_data)
        self.data['activation_center_movement_angle_velocity_std'] = np.nanstd(activation_center_movement_angle_velocity_data)

        self.data['activation_mass_asymmetry'] = activation_mass_asymmetry_data
        self.data['activation_mass_asymmetry_mean'] = np.nanmean(activation_mass_asymmetry_data)
        self.data['activation_mass_asymmetry_std'] = np.nanstd(activation_mass_asymmetry_data)

        self.data['activation_mass_distribution'] = activation_mass_distribution_data
        self.data['activation_mass_distribution_mean'] = np.nanmean(activation_mass_distribution_data)
        self.data['activation_mass_distribution_std'] = np.nanstd(activation_mass_distribution_data)

        self.data['activation_hu1'] = activation_hu1_data
        self.data['activation_hu1_mean'] = np.nanmean(activation_hu1_data)
        self.data['activation_hu1_std'] = np.nanstd(activation_hu1_data)

        self.data['activation_hu2'] = activation_hu2_data
        self.data['activation_hu2_mean'] = np.nanmean(activation_hu2_data)
        self.data['activation_hu2_std'] = np.nanstd(activation_hu2_data)

        self.data['activation_hu3'] = activation_hu3_data
        self.data['activation_hu3_mean'] = np.nanmean(activation_hu3_data)
        self.data['activation_hu3_std'] = np.nanstd(activation_hu3_data)

        self.data['activation_hu4'] = activation_hu4_data
        self.data['activation_hu4_mean'] = np.nanmean(activation_hu4_data)
        self.data['activation_hu4_std'] = np.nanstd(activation_hu4_data)

        self.data['activation_hu5'] = activation_hu5_data
        self.data['activation_hu5_mean'] = np.nanmean(activation_hu5_data)
        self.data['activation_hu5_std'] = np.nanstd(activation_hu5_data)

        self.data['activation_hu6'] = activation_hu6_data
        self.data['activation_hu6_mean'] = np.nanmean(activation_hu6_data)
        self.data['activation_hu6_std'] = np.nanstd(activation_hu6_data)

        self.data['activation_hu7'] = activation_hu7_data
        self.data['activation_hu7_mean'] = np.nanmean(activation_hu7_data)
        self.data['activation_hu7_std'] = np.nanstd(activation_hu7_data)

        self.data['activation_hu8'] = activation_hu8_data
        self.data['activation_hu8_mean'] = np.nanmean(activation_hu8_data)
        self.data['activation_hu8_std'] = np.nanstd(activation_hu8_data)

        self.data['activation_flusser9'] = activation_flusser9_data
        self.data['activation_flusser9_mean'] = np.nanmean(activation_flusser9_data)
        self.data['activation_flusser9_std'] = np.nanstd(activation_flusser9_data)

        self.data['activation_flusser10'] = activation_flusser10_data
        self.data['activation_flusser10_mean'] = np.nanmean(activation_flusser10_data)
        self.data['activation_flusser10_std'] = np.nanstd(activation_flusser10_data)

        self.data['activation_flusser11'] = activation_flusser11_data
        self.data['activation_flusser11_mean'] = np.nanmean(activation_flusser11_data)
        self.data['activation_flusser11_std'] = np.nanstd(activation_flusser11_data)

        self.data['activation_flusser12'] = activation_flusser12_data
        self.data['activation_flusser12_mean'] = np.nanmean(activation_flusser12_data)
        self.data['activation_flusser12_std'] = np.nanstd(activation_flusser12_data)

        self.data['activation_flusser13'] = activation_flusser13_data
        self.data['activation_flusser13_mean'] = np.nanmean(activation_flusser13_data)
        self.data['activation_flusser13_std'] = np.nanstd(activation_flusser13_data)

        self.data['positive_growth_mass'] = positive_growth_mass_data
        self.data['positive_growth_mass_mean'] = np.nanmean(positive_growth_mass_data)
        self.data['positive_growth_mass_std'] = np.nanstd(positive_growth_mass_data)

        self.data['positive_growth_volume'] = positive_growth_volume_data
        self.data['positive_growth_volume_mean'] = np.nanmean(positive_growth_volume_data)
        self.data['positive_growth_volume_std'] = np.nanstd(positive_growth_volume_data)

        self.data['positive_growth_density'] = positive_growth_density_data
        self.data['positive_growth_density_mean'] = np.nanmean(positive_growth_density_data)
        self.data['positive_growth_density_std'] = np.nanstd(positive_growth_density_data)

        self.data['positive_growth_center_position'] = positive_growth_center_position_data

        self.data['positive_growth_center_velocity'] = positive_growth_center_velocity_data
        self.data['positive_growth_center_velocity_mean'] = np.nanmean(positive_growth_center_velocity_data)
        self.data['positive_growth_center_velocity_std'] = np.nanstd(positive_growth_center_velocity_data)

        self.data['positive_growth_center_movement_angle'] = positive_growth_center_movement_angle_data
        self.data['positive_growth_center_movement_angle_mean'] = ad.helper.statistics.nan_mean_over_angles_degrees(positive_growth_center_movement_angle_data)
        #self.data['positive_growth_center_movement_angle_std'] = np.nanstd(positive_growth_center_movement_angle_data)

        self.data['positive_growth_center_movement_angle_velocity'] = positive_growth_center_movement_angle_velocity_data
        self.data['positive_growth_center_movement_angle_velocity_mean'] = np.nanmean(positive_growth_center_movement_angle_velocity_data)
        self.data['positive_growth_center_movement_angle_velocity_std'] = np.nanstd(positive_growth_center_movement_angle_velocity_data)

        self.data['activation_positive_growth_centroid_distance'] = activation_positive_growth_centroid_distance_data
        self.data['activation_positive_growth_centroid_distance_mean'] = np.nanmean(activation_positive_growth_centroid_distance_data)
        self.data['activation_positive_growth_centroid_distance_std'] = np.nanstd(activation_positive_growth_centroid_distance_data)


    @staticmethod
    def calc_distance_matrix(size_y, size_x):

        dist_mat = np.zeros([size_y, size_x])

        mid_y = (size_y - 1)/ 2
        mid_x = (size_x - 1)/ 2
        mid = np.array([mid_y, mid_x])

        max_dist = int(np.linalg.norm(mid))

        for y in range(size_y):
            for x in range(size_x):
                dist_mat[y][x] = (1 - int(np.linalg.norm(mid - np.array([y, x]))) / max_dist) ** LeniaStatistics.DISTANCE_WEIGHT

        return dist_mat


    @staticmethod
    def center_of_mass(input_array):

        center = np.array(ndimage.measurements.center_of_mass(input_array))

        if np.any(np.isnan(center)):
            center = np.array([int((input_array.shape[0] - 1) / 2), int((input_array.shape[1] - 1) / 2)])

        return center



class LeniaAnimalExplorer(ad.core.Explorer):
    '''
    Explores existing animals for the lenia system that were discovered by Bert Chan and are described in the file leniaanimals.json.

    Most functions are adapted from the original Lenia code by Bert Chan.
    '''

    @staticmethod
    def default_config():
        def_config = ad.core.Explorer.default_config()

        py_script_file_folder = os.path.dirname(os.path.realpath(__file__))

        def_config.animals_source_file = os.path.join(py_script_file_folder, 'leniaanimals.json')
        def_config.num_of_steps = 300
        def_config.default_run_parameters = {'kn': 1, 'gn': 1}

        return def_config


    def run(self, animal_ids=None, verbose=True):

        system_size = (self.system.system_parameters.size_y, self.system.system_parameters.size_x)
        system_middle_point = (int(system_size[0]/2), int(system_size[1]/2))

        #################################################
        # load animal data from json file

        with open(self.config.animals_source_file, encoding='utf-8') as file:
            animal_data = json.load(file)

        animal_configs = dict()
        animal_id = 0
        for animal_data_entry in animal_data:

            # only entries with a params field encode an animal
            if 'params' in animal_data_entry:

                animal_config = dict()

                animal_config['id'] = animal_id
                animal_config['name'] = animal_data_entry['name']

                params_run = {**self.config.default_run_parameters, **animal_data_entry['params']}

                # b can be a vector described  as string, e.g.: [0.5,2] = '1/2,2'
                params_run['b'] = [float(fractions.Fraction(st)) for st in params_run['b'].split(',')]

                # load init state from the description string
                animal_array = rle2arr(animal_data_entry['cells'])

                animal_height = animal_array.shape[0]
                animal_width = animal_array.shape[1]

                if animal_height > system_size[0] or animal_width > system_size[1]:
                    warnings.warn('Animal {} (id = {}) is not loaded because it is too big ([{}, {}]).'.format(animal_data_entry['name'],
                                                                                                               animal_id,
                                                                                                               animal_array.shape[0],
                                                                                                               animal_array.shape[1]), Warning)

                else:
                    # load the animal
                    init_cond = np.zeros(system_size)

                    y = int(system_middle_point[0] - animal_height / 2)
                    x = int(system_middle_point[1] - animal_width / 2)

                    init_cond[y:y+animal_height, x:x+animal_width] = animal_array
                    params_run['init_state'] = init_cond

                    animal_config['run_parameters'] = params_run

                    animal_configs[animal_id] = animal_config

                animal_id = animal_id + 1

        #############################################
        # explore the animals

        if animal_ids is None:
            animal_ids = animal_configs.keys()

        elif not isinstance(animal_ids, collections.Iterable):
            # if the given animal if is not a list, then encapsulate it
            animal_ids = [animal_ids]

        counter = 0

        if verbose:
            ad.gui.print_progress_bar(counter, len(animal_ids), 'Explorations: ')

        for animal_id in animal_ids:
            counter = counter + 1

            if animal_id not in self.data:
                animal_config = animal_configs[animal_id]

                # run the experiment
                # run the experiment
                observations, statistics = self.system.run(run_parameters=animal_config['run_parameters'],
                                                           stop_conditions=self.config.num_of_steps)

                self.data.add_run_data(id=animal_id,
                                       name=animal_config['name'],
                                       run_parameters=animal_config['run_parameters'],
                                       observations=observations,
                                       statistics=statistics)

            if verbose:
                ad.gui.print_progress_bar(counter, len(animal_ids), 'Explorations: ')