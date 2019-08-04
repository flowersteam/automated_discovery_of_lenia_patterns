import torch
import numpy as np
import autodisc as ad


def neat_to_torch_actfunc(neat_func):

    # some functions such as exp, etc are also implemented directly in torch
    if hasattr(torch, neat_func):
        return getattr(torch, neat_func)

    # other functions need to be specified
    if neat_func == 'delphineat_gauss':
       return ad.cppn.activations.delphineat_gauss_torch_activation

    if neat_func == 'delphineat_sigmoid':
       return ad.cppn.activations.delphineat_sigmoid_torch_activation

    raise ValueError('Unknown activation function {!r}!', neat_func)


class RecurrentCPPN(torch.nn.Module):

    # TODO: neat allows also for different aggregation function!

    def __init__(self, input_neurons, neurons, output_neurons, connections, biases, responses, activations, dtype=torch.float32):
        """

        input_neurons = [id, ...]
        neurons = [id, ...]
        output_neurons = [id, ...]
        connections = [[from_id, to_id, weight], ...]

        """
        super().__init__()

        self.dtype = dtype

        self.n_inputs = len(input_neurons)
        self.n_outputs = len(output_neurons)

        self.output_neuron_idxs = len(output_neurons)

        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        self.responses = torch.nn.ParameterList()

        self.activations = []

        # holds the idxs of all the neurons that are connected to a neuron
        self.input_neuron_idxs = []

        # neurons are later identified by the index of this list
        all_neurons = np.hstack((input_neurons, neurons, output_neurons))
        not_input_neurons = np.hstack((neurons, output_neurons))

        self.output_neuron_idxs = list(range(len(neurons), len(not_input_neurons)))

        # iterate over hidden neurons and output neurons and create for each a tensor representing its input weights
        for neuron_id in not_input_neurons:

            # get the idxs of the neurons from which it gets inputs
            if len(connections) > 0:
                neuron_connections = connections[connections[:, 1] == neuron_id]

                # find the idx (used by this pytorch class) for each input neuron based on its id
                neuron_to_neuron_idxs = [np.where(all_neurons == x)[0][0] for x in neuron_connections[:, 0]]
            else:
                neuron_to_neuron_idxs = []

            self.input_neuron_idxs.append(neuron_to_neuron_idxs)

            if len(neuron_to_neuron_idxs) == 0:
                # no connections into the neuron, so no weights exist and also the response is not necessary
                w = torch.tensor([], dtype=self.dtype)
                r = torch.tensor([], dtype=self.dtype)
            else:
                w = torch.tensor([neuron_connections[:, 2]], dtype=self.dtype).t()
                r = torch.tensor([[responses[neuron_id]]], dtype=self.dtype)

            self.weights.append(torch.nn.Parameter(w))
            self.responses.append(torch.nn.Parameter(r))

            b = torch.tensor([[biases[neuron_id]]], dtype=self.dtype)
            self.biases.append(torch.nn.Parameter(b))

            self.activations.append(activations[neuron_id])


    def forward(self, input, prev_state):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        new_state = prev_state.clone()

        # put activations of the input and the other neurons in a  single tensor
        all_activation = torch.cat((input, new_state), 1)

        # go over neurons and compute their output, based on the previous state of the neurons
        for neuron_idx in range(len(self.input_neuron_idxs)):

            # get connection weights
            weights = self.weights[neuron_idx]

            if weights.shape[0] > 0:
                # do not regard input for neurons without ingoing connections

                input_act = all_activation[:, self.input_neuron_idxs[neuron_idx]]

                # compute new activation( = act(bias + response * sum(inputs * weights)))
                new_state[:, neuron_idx] = self.activations[neuron_idx](self.biases[neuron_idx] + self.responses[neuron_idx] * input_act.mm(weights)).t()

            else:
                # only regard the bias for the neurons without ingoing connections
                new_state[:, neuron_idx] = self.activations[neuron_idx](self.biases[neuron_idx]).t()

        return new_state[:, self.output_neuron_idxs], new_state


    def init_state(self, n=1):
        '''Initial state for the forward functionj'''
        return torch.zeros((n,len(self.input_neuron_idxs)), dtype=self.dtype)


    def activate(self, input, repetitions=1):
        '''
        Computes network output.

        Note: Does not compute gradients.

        :param input: [n_data * n_inputs] matrix.
        :param repetitions: Number of repetitions the network should be executed.
        :return: Numpy array with output ([n_data * n_outputs])
        '''

        # if input has only one dimension, assume it is only one data point and create
        input_dim = len(np.shape(input))

        if input_dim > 1: # 2d array
            net_input = torch.tensor(input, dtype=self.dtype) # TODO: check if this copies the data from the array or uses the same memory, want to use same memory

        elif input_dim == 1: # 1d array
            net_input = torch.tensor([input], dtype=self.dtype)

        else: # scalar value
            net_input = torch.tensor([[input]], dtype=self.dtype)

        if net_input.shape[1] != self.n_inputs:
            raise ValueError('Wrong input dimension! Given: {!r}, Network expects: {!r}'.format(net_input.shape[1], self.n_inputs))

        net_state = self.init_state(net_input.shape[0])

        output = None
        with torch.no_grad():
            for _ in range(repetitions):
                [output, net_state] = self.forward(net_input, net_state)

        # if only 1-d vector was given, then return 1-d vector
        if input_dim == 1:
            output = output[0]
        elif input_dim ==0:  # scalar value
            output = output[0][0]

        return output.numpy()



    @staticmethod
    def create_from_genome(genome, neat_config, dtype=torch.float64):

        genome_config = neat_config.genome_config

        input_neurons = list(reversed(range(-genome_config.num_inputs,0)))
        output_neurons = list(range(genome_config.num_outputs)) # TODO: check if correct

        neurons = []
        connections = []
        biases = dict()
        responses = dict()
        activations = dict()

        for node_id, node in genome.nodes.items():

            if node.aggregation != 'sum':
                raise NotImplementedError('Only sum - aggregation function implemented for PyTorch networks!')

            if node_id not in output_neurons:
                neurons.append(node_id)

            biases[node_id] = node.bias
            responses[node_id] = node.response
            activations[node_id] = neat_to_torch_actfunc(node.activation)

        for (from_id, to_id), connection in genome.connections.items():
            if connection.enabled:
                connections.append([from_id, to_id, connection.weight])

        connections = np.array(connections)

        return RecurrentCPPN(input_neurons, neurons, output_neurons, connections, biases, responses, activations, dtype)
