from nn_modeller.Overall_Base_Network import Overall_Base_Network

class Base_Network(Overall_Base_Network):
    def __init__(self, input_dim, layers_info, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed):
        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()
        super().__init__(input_dim, layers_info, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed)
        self.initialiser = initialiser

    def create_str_to_activations_converter(self):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"elu": "nn.ELU", "hardshrink": "nn.Hardshrink", "hardtanh": "nn.Hardtanh",
                                        "leakyrelu": "nn.LeakyReLU", "logsigmoid": "nn.LogSigmoid", "prelu": "nn.PReLU",
                                        "relu": "nn.ReLU", "relu6": "nn.ReLU6", "rrelu": "nn.RReLU", "selu": "nn.SELU",
                                        "sigmoid": "nn.Sigmoid", "softplus": "nn.Softplus",
                                        "logsoftmax": "nn.LogSoftmax",
                                        "softshrink": "nn.Softshrink", "softsign": "nn.Softsign", "tanh": "nn.Tanh",
                                        "tanhshrink": "nn.Tanhshrink", "softmin": "nn.Softmin",
                                        "softmax": "nn.Softmax",
                                        "none": "default"}
        return str_to_activations_converter

    def create_str_to_initialiser_converter(self):
        """Creates a dictionary which converts strings to initialiser"""
        str_to_initialiser_converter = {"uniform": "nn.init.uniform_", "normal": "nn.init.normal_",
                                        "eye": "nn.init.eye_",
                                        "xavier_uniform": "nn.init.xavier_uniform_", "xavier": "nn.init.xavier_uniform_",
                                        "xavier_normal": "nn.init.xavier_normal_",
                                        "kaiming_uniform": "nn.init.kaiming_uniform_",
                                        "kaiming": "nn.init.kaiming_uniform_",
                                        "kaiming_normal": "nn.init.kaiming_normal_", "he": "nn.init.kaiming_normal_",
                                        "orthogonal": "nn.init.orthogonal_", "default": "default"}
        return str_to_initialiser_converter

    def create_dropout_layer(self):
        return [f'self.dropout = nn.Dropout(p = {self.dropout})']


    def create_output_layers(self):
        output_layers = []
        network_type = type(self).__name__
        if network_type in ["CNN", "RNN"]:
            if not isinstance(self.layers_info[-1][0], list): self.layers_info[-1] = [self.layers_info[-1]]
        elif network_type == "NN":
            if isinstance(self.layers_info[-1], int) and isinstance(self.layers_info[-2], int):
                self.layers_info[-1] = [self.layers_info[-2], self.layers_info[-1]]
        else:
            raise ValueError("Network type not recognised")
        layers = [0, self.layers_info[-1][0]]
        for layer_ix, layer in enumerate(self.layers_info[-1][1:]):
            activation = self.get_activation(self.output_activation, layer_ix)
            initialization = self.get_initialization(self.initialiser, len(self.layers_info) - 1)
            layers[0], layers[1] = layers[1], layer
            self.batch_norm = False
            self.dropout = 0
            self.create_and_append_layer(layers, output_layers, activation, output_layer=True, initializer = initialization)
        return output_layers

    def create_hidden_layers(self):
        hidden_layers = []
        layers = [0, self.input_dim]
        for layer_ix, layer in enumerate(self.layers_info[:-1]):
            activation = self.get_activation(self.hidden_activations, layer_ix)
            initialization = self.get_initialization(self.initialiser, layer_ix)
            layers[0], layers[1] = layers[1], layer
            self.create_and_append_layer(layers, hidden_layers, activation, output_layer=False, initializer = initialization)
        return hidden_layers

    def create_batch_norm_layers(self):
        batch_norm_layers = []
        bnl_count = 1
        for layer_ix, layer in enumerate(self.layers_info[:-1]):
            batch_norm_layers.extend([f'self.bn_{bnl_count} = nn.BatchNormld({layer})'])
            bnl_count += 1
        return batch_norm_layers

    def print_model_summary(self):
        print(self)
        print(f'class Model(nn.Module):')
        print(f'\tdef __init__(self):')
        print(f'\t\tnn.Module.__init__(self)')
        for layer in self.hidden_layers:
            print(f'\t\t{layer}')
        for layer in self.output_layers:
            print(f'\t\t{layer}')
        for layer in self.dropout_layer:
            print(f'\t\t{layer}')
        for layer in self.batch_norm_layers:
            print(f'\t\t{layer}')
        for layer in self.initialization:
            print(f'\t\t{layer}')
        print(f'\tdef forward(self, x):')
        for layer in self.sum_up:
            print(f'\t\t{layer}')
        print(f'\t\treturn x')

    def get_model_summary(self):
        str = ''
        str += 'class Model(nn.Module):\n'
        str += '\tdef __init__(self):\n'
        str += '\t\tnn.Module.__init__(self)\n'
        for layer in self.hidden_layers:
            str += f'\t\t{layer}\n'
        for layer in self.output_layers:
            str += f'\t\t{layer}\n'
        for layer in self.dropout_layer:
            str += f'\t\t{layer}\n'
        for layer in self.batch_norm_layers:
            str += f'\t\t{layer}\n'
        for layer in self.initialization:
            str += f'\t\t{layer}\n'
        str += '\tdef forward(self, x):\n'
        for layer in self.sum_up:
            str += f'\t\t{layer}\n'
        str += '\t\treturn x'
        return  str