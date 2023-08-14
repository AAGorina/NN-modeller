from nn_modeller.tensorflow.Base_Network import Base_Network

class NN(Base_Network):
    def __init__(self, layers_info, output_activation=None, hidden_activations="relu", dropout=0.0, initialiser="glorot_uniform",
                 batch_norm=False, y_range= (), random_seed=0,
                 input_dim=None):

        self.layer_count_dense = 1

        Base_Network.__init__(self, layers_info, output_activation, hidden_activations, dropout, initialiser,
                              batch_norm, y_range, random_seed, input_dim)

    def check_all_user_inputs_valid(self):
        self.check_NN_layers_valid()
        self.check_activations_valid()
        self.check_initialiser_valid()

    def create_and_append_layer(self, layer, list_to_append_layer_to,activation=None, initialisation="initializers.glorot_uniform", output_layer=False):
        list_to_append_layer_to.extend([f'self.liner_{self.layer_count_dense} = Dense({layer}, activation = "{activation}", '
                                        f'kernel_initializer = {initialisation})'])

        tmp = f'self.liner_{self.layer_count_dense}(x)'
        if self.batch_norm:
            tmp = f'self.batch_norm({tmp}, training = training)'
        if self.dropout != 0:
            tmp = f'self.dropout({tmp})'
        self.sum_up.extend([f'x = {tmp}'])
        self.layer_count_dense += 1
        self.last_layer_type = 'linear'
