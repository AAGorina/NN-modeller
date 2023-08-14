import numpy as np
from nn_modeller.pytorch.Base_Network import Base_Network

class NN(Base_Network):
    def __init__(self, input_dim, layers_info, output_activation=None,
                 hidden_activations="relu", dropout=0.0, initialiser="default", batch_norm=False,
                 y_range= (), random_seed=0):

        self.layer_count_linear = 1
        self.input_dim = input_dim
        self.initialiser = initialiser
        Base_Network.__init__(self, input_dim, layers_info, output_activation, hidden_activations, dropout, initialiser,
                              batch_norm, y_range, random_seed)

    def check_all_user_inputs_valid(self):
        self.check_NN_input_dim_valid()
        self.check_NN_layers_valid()
        self.check_activations_valid()
        self.check_initialiser_valid()

    def create_and_append_layer(self, layer, list_to_append_layer_to, activation="default", output_layer=False, initializer = "default"):
        list_to_append_layer_to.extend([f'self.linear_{self.layer_count_linear} = nn.Linear({layer[0]}, {layer[1]})'])

        tmp = f'self.liner_{self.layer_count_linear}(x)'
        if activation != "default":
            tmp = f'{activation}({tmp})'
        if self.batch_norm:
            tmp = f'self.bn_{self.layer_count_linear}({tmp})'
        if self.dropout != 0:
            tmp = f'self.dropout({tmp})'
        self.sum_up.extend([f'x = {tmp}'])
        if initializer != "default":
            self.initialization.extend([f'{initializer}(self.linear_{self.layer_count_linear}.weight)'])
        self.layer_count_linear += 1
        self.last_layer_type = 'linear'
