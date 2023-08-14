from nn_modeller.Overall_Base_Network import Overall_Base_Network

class Base_Network(Overall_Base_Network):
    def __init__(self, layers_info, output_activation, hidden_activations, dropout, initialiser, batch_norm, y_range,
                 random_seed, input_dim):
        if input_dim is not None: print("You don't need to provide input_dim for a tensorflow network")
        super().__init__(None, layers_info, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed)

    def create_str_to_activations_converter(self):
        str_to_activations_converter = {"elu": "elu", "exponential": "exponential",
                                        "hard_sigmoid": "hard_sigmoid", "linear": "linear",
                                        "relu": "relu", "selu": "selu",
                                        "sigmoid": "sigmoid",
                                        "softmax": "softmax", "softplus": "softplus",
                                        "softsign": "softsign", "tanh": "tanh",
                                        "none": "linear"}
        return str_to_activations_converter

    def create_str_to_initialiser_converter(self):
        str_to_initialiser_converter = {"glorot_normal": "initializers.glorot_normal", "glorot_uniform": "initializers.glorot_uniform",
                                         "he_normal": "initializers.he_normal()",
                                        "he_uniform": "initializers.he_uniform()", "lecun_normal": "initializers.lecun_normal()",
                                        "lecun_uniform": "initializers.lecun_uniform()", "truncated_normal": "initializers.TruncatedNormal",
                                        "variance_scaling": "initializers.VarianceScaling", "default": "initializers.glorot_uniform"}
        return str_to_initialiser_converter

    def create_dropout_layer(self):
        return [f'self.dropout = tf.keras.layers.Dropout(rate = {self.dropout})']

    def create_hidden_layers(self):
        hidden_layers = []
        for layer_ix, layer in enumerate(self.layers_info[:-1]):
            activation = self.get_activation(self.hidden_activations, layer_ix)
            initialisation = self.get_initialization(self.initialiser, layer_ix)
            self.create_and_append_layer(layer, hidden_layers, activation, initialisation, output_layer=False)
        return hidden_layers

    def create_output_layers(self):
        output_layers = []
        network_type = type(self).__name__
        if network_type in ["CNN"]:
            if not isinstance(self.layers_info[-1][0], list): self.layers_info[-1] = [self.layers_info[-1]]
        elif network_type == "NN":
            if isinstance(self.layers_info[-1], int): self.layers_info[-1] = [self.layers_info[-1]]
        else:
            raise ValueError("Network type not recognised")
        for output_layer_ix, output_layer in enumerate(self.layers_info[-1]):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            initialisation = self.get_initialization(self.initialiser, len(self.layers_info)-1)
            self.batch_norm = False
            self.dropout = 0
            self.create_and_append_layer(output_layer, output_layers, activation, initialisation, output_layer=True)
        return output_layers

    def create_batch_norm_layers(self):
        batch_norm_layers = []
        if self.batch_norm:
            batch_norm_layers.extend([f'self.batch_norm = tf.keras.layers.BatchNormalization()'])
        return batch_norm_layers

    def print_model_summary(self):
        print(f'class model(Model):')
        print(f'\tdef __init__(self):')
        print(f'\t\tModel.__init__(self)')
        for layer in self.hidden_layers:
            print(f'\t\t{layer}')
        for layer in self.output_layers:
            print(f'\t\t{layer}')
        for layer in self.dropout_layer:
            print(f'\t\t{layer}')
        for layer in self.batch_norm_layers:
            print(f'\t\t{layer}')
        print(f'def call(self, x, training = False):')
        for layer in self.sum_up:
            print(f'\t\t{layer}')
        print(f'\t\treturn x')

    def get_model_summary(self):
        str = ''
        str += 'class model(Model):\n' \
               '\tdef __init__(self):\n' \
               '\t\tModel.__init__(self)\n'
        for layer in self.hidden_layers:
            str += f'\t\t{layer}\n'
        for layer in self.output_layers:
            str += f'\t\t{layer}\n'
        for layer in self.dropout_layer:
            str += f'\t\t{layer}\n'
        for layer in self.batch_norm_layers:
            str += f'\t\t{layer}\n'
        str += 'def call(self, x, training = False):\n'
        for layer in self.sum_up:
            str += f'\t\t{layer}\n'
        str += '\t\treturn x'
        return str