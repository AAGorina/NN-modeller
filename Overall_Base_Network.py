class Overall_Base_Network():

    def __init__(self, input_dim, layers_info, output_activation, hidden_activations, dropout, initialiser, batch_norm,
                 y_range, random_seed):

        self.input_dim = input_dim
        self.layers_info = layers_info

        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.dropout = dropout
        self.initialiser = initialiser
        self.batch_norm = batch_norm
        self.y_range = y_range

        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()

        self.check_all_user_inputs_valid()

        self.initialization = []

        self.sum_up = []
        self.last_layer_type = ''
        self.hidden_layers = self.create_hidden_layers()

        self.batch_norm_layers = []
        if self.batch_norm: self.batch_norm_layers = self.create_batch_norm_layers()

        self.dropout_layer = []
        if self.dropout != 0:
            self.dropout_layer = self.create_dropout_layer()

        self.output_layers = self.create_output_layers()

    def check_NN_layers_valid(self):
        assert isinstance(self.layers_info, list), "hidden_units must be a list"
        list_error_msg = "neurons must be a list of integers"
        integer_error_msg = "Every element of hidden_units must be 1 or higher"
        activation_error_msg = "The number of output activations provided should match the number of output layers"
        for neurons in self.layers_info[:-1]:
            assert isinstance(neurons, int), list_error_msg
            assert neurons > 0, integer_error_msg
        output_layer = self.layers_info[-1]
        if isinstance(output_layer, list):
            assert len(output_layer) == len(self.output_activation), activation_error_msg
            for output_dim in output_layer:
                assert isinstance(output_dim, int), list_error_msg
                assert output_dim > 0, integer_error_msg
        else:
            assert isinstance(self.output_activation, str) or self.output_activation is None, activation_error_msg
            assert isinstance(output_layer, int), list_error_msg
            assert output_layer > 0, integer_error_msg

    def check_NN_input_dim_valid(self):
        assert isinstance(self.input_dim, int), "input_dim must be an integer"
        assert self.input_dim > 0, "input_dim must be 1 or higher"

    def check_activations_valid(self):
        valid_activations_strings = self.str_to_activations_converter.keys()
        if self.output_activation is None: self.output_activation = "None"
        if isinstance(self.output_activation, list):
            for activation in self.output_activation:
                if activation is not None:
                    assert activation.lower() in set(valid_activations_strings), "Output activations must be string from list {}".format(valid_activations_strings)
        else:
            assert self.output_activation.lower() in set(valid_activations_strings), "Output activation {} must be string from list {}".format(self.output_activation, valid_activations_strings)
        assert isinstance(self.hidden_activations, str) or isinstance(self.hidden_activations, list), "hidden_activations must be a string or a list of strings"
        if isinstance(self.hidden_activations, str):
            assert self.hidden_activations.lower() in set(valid_activations_strings), "hidden_activations must be from list {}".format(valid_activations_strings)
        elif isinstance(self.hidden_activations, list):
            assert len(self.hidden_activations) == len(self.layers_info), "if hidden_activations is a list then you must provide 1 activation per hidden layer"
            for activation in self.hidden_activations:
                assert isinstance(activation, str), "hidden_activations must be a string or list of strings"
                assert activation.lower() in set(valid_activations_strings), "each element in hidden_activations must be from list {}".format(valid_activations_strings)

    def check_initialiser_valid(self):
        valid_initialisers = set(self.str_to_initialiser_converter.keys())
        if isinstance(self.initialiser, str):
            assert self.initialiser.lower() in set(valid_initialisers), "initialiser must be from list {}".format(valid_initialisers)
        elif isinstance(self.initialiser, list):
            assert len(self.initialiser) == len(self.layers_info), "if initialiser is a list then you must provide 1 activation per hidden layer"
            for init in self.initialiser:
                assert isinstance(init, str), "initialiser must be a string or list of strings"
                assert init.lower() in set(valid_initialisers), "each element in initialiser must be from list {}".format(valid_initialisers)

    def get_initialization (self, initializer, ix = None):
        if isinstance(initializer, list):
            return self.str_to_initialiser_converter[str(initializer[ix]).lower()]
        return self.str_to_initialiser_converter[str(initializer).lower()]

    def get_activation(self, activations, ix=None):
        if isinstance(activations, list):
            return self.str_to_activations_converter[str(activations[ix]).lower()]
        return self.str_to_activations_converter[str(activations).lower()]
