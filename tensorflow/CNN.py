from nn_modeller.tensorflow.Base_Network import Base_Network

class CNN(Base_Network):
    def __init__(self, layers_info, output_activation=None, hidden_activations="relu", dropout= 0.0, initialiser="default",
                 batch_norm=False, y_range=(), random_seed=0, input_dim=None):
        self.valid_cnn_hidden_layer_types = {'conv', 'maxpool', 'avgpool', 'linear'}

        self.linear_count = 1
        self.conv_count = 1
        self.maxp_count = 1
        self. avgp_count = 1

        Base_Network.__init__(self, layers_info, output_activation, hidden_activations, dropout, initialiser,
                              batch_norm, y_range, random_seed, input_dim)

    def check_all_user_inputs_valid(self):
        self.check_CNN_layers_valid()
        self.check_activations_valid()
        self.check_initialiser_valid()

    def check_CNN_layers_valid(self):
        error_msg_layer_type = "First element in a layer specification must be one of {}".format(self.valid_cnn_hidden_layer_types)
        error_msg_conv_layer = """Conv layer must be of form ['conv', channels, kernel_size, stride, padding] where the 
                               variables are all non-negative integers except padding which must be either "valid" or "same"""
        error_msg_maxpool_layer = """Maxpool layer must be of form ['maxpool', kernel_size, stride, padding] where the 
                               variables are all non-negative integers except padding which must be either "valid" or "same"""
        error_msg_avgpool_layer = """Avgpool layer must be of form ['avgpool', kernel_size, stride, padding] where the 
                               variables are all non-negative integers except padding which must be either "valid" or "same"""
        error_msg_linear_layer = """Linear layer must be of form ['linear', out] where out is a non-negative integers"""
        assert isinstance(self.layers_info, list), "layers must be a list"

        all_layers = self.layers_info[:-1]
        output_layer = self.layers_info[-1]
        assert isinstance(output_layer, list), "layers must be a list"
        if isinstance(output_layer[0], list):
            assert len(output_layer) == len(self.output_activation), "Number of output activations must equal number of output heads"
            for layer in output_layer:
                all_layers.append(layer)
                assert isinstance(layer[0], str), error_msg_layer_type
                assert layer[0].lower() == "linear", "Final layer must be linear"
        else:
            all_layers.append(output_layer)
            assert isinstance(output_layer[0], str), error_msg_layer_type
            assert output_layer[0].lower() == "linear", "Final layer must be linear"

        for layer in all_layers:
            assert isinstance(layer, list), "Each layer must be a list"
            assert isinstance(layer[0], str), error_msg_layer_type
            layer_type_name = layer[0].lower()
            assert layer_type_name in self.valid_cnn_hidden_layer_types, "Layer name {} not valid, use one of {}".format(layer_type_name, self.valid_cnn_hidden_layer_types)
            if layer_type_name == "conv":
                assert len(layer) == 5, error_msg_conv_layer
                for ix in range(3): assert isinstance(layer[ix+1], int) and layer[ix+1] > 0, error_msg_conv_layer
                assert isinstance(layer[4], str) and layer[4].lower() in ["valid", "same"], error_msg_conv_layer
            elif layer_type_name == "maxpool":
                assert len(layer) == 4, error_msg_maxpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_maxpool_layer
                if layer[1] != layer[2]: print("NOTE that your maxpool kernel size {} isn't the same as your stride {}".format(layer[1], layer[2]))
                assert isinstance(layer[3], str) and layer[3].lower() in ["valid", "same"], error_msg_maxpool_layer
            elif layer_type_name == "avgpool":
                assert len(layer) == 4, error_msg_avgpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_avgpool_layer
                assert isinstance(layer[3], str) and layer[3].lower() in ["valid", "same"], error_msg_avgpool_layer
                if layer[1] != layer[2]:print("NOTE that your avgpool kernel size {} isn't the same as your stride {}".format(layer[1], layer[2]))
            elif layer_type_name == "linear":
                assert len(layer) == 2, error_msg_linear_layer
                for ix in range(1): assert isinstance(layer[ix+1], int) and layer[ix+1] > 0
            else:
                raise ValueError("Invalid layer name")

        rest_must_be_linear = False
        for ix, layer in enumerate(all_layers):
            if rest_must_be_linear: assert layer[0].lower() == "linear", "If have linear layers then they must come at end"
            if layer[0].lower() == "linear":
                rest_must_be_linear = True

    def create_and_append_layer(self, layer, list_to_append_layer_to, activation=None, initialisation = 'initializers.glorot_uniform',  output_layer=False):
        layer_name = layer[0].lower()
        assert layer_name in self.valid_cnn_hidden_layer_types, "Layer name {} not valid, use one of {}".format(
            layer_name, self.valid_cnn_hidden_layer_types)
        if layer_name == "conv":
            """list_to_append_layer_to.extend([Conv2D(filters=layer[1], kernel_size=layer[2],
                                                strides=layer[3], padding=layer[4], activation=activation,
                                                   kernel_initializer=self.initialiser_function)])"""
            list_to_append_layer_to.extend([f'self.conv2_{self.conv_count} = Conv2D(filters = {layer[1]}, '
                                            f'kernel_size = {layer[2]}, strides = {layer[3]}, '
                                            f'padding = "{layer[4]}", activation = "{activation}", '
                                            f'kernel_initializer = {initialisation})'])

            tmp = f'self.conv2_{self.conv_count}(x)'
            if self.batch_norm:
                tmp = f'self.batch_norm({tmp}, training = training)'

            self.sum_up.extend([f'x = {tmp}'])

            self.conv_count += 1
        elif layer_name == "maxpool":
            """list_to_append_layer_to.extend([MaxPool2D(pool_size=(layer[1], layer[1]),
                                                   strides=(layer[2], layer[2]), padding=layer[3])])"""
            list_to_append_layer_to.extend([f'self.maxpool_{self.maxp_count} = MaxPool2D(pool_size=({layer[1]}, {layer[1]}), '
                                            f'strides=({layer[2]}, {layer[2]}), padding = "{layer[3]}")'])

            tmp = f'self.maxpool_{self.maxp_count}(x)'
            if self.dropout != 0:
                tmp = f'self.dropout({tmp})'
            self.sum_up.extend([f'x = {tmp}'])

            self.maxp_count += 1
        elif layer_name == "avgpool":
            """list_to_append_layer_to.extend([AveragePooling2D(pool_size=(layer[1], layer[1]),
                                                   strides=(layer[2], layer[2]), padding=layer[3])])"""
            list_to_append_layer_to.extend(
                [f'self.avgpool_{self.avgp_count} = AveragePooling2D(pool_size=({layer[1]}, {layer[1]}), '
                 f'strides=({layer[2]}, {layer[2]}), padding = "{layer[3]}")'])

            tmp = f'self.avgpool_{self.avgp_count}(x)'

            if self.dropout != 0:
                tmp = f'self.dropout({tmp})'
            self.sum_up.extend([f'x = {tmp}'])
            self.avgp_count += 1
        elif layer_name == "linear":
            if self.last_layer_type != "linear":
                self.sum_up.extend([f'x = Flatten()(x)'])
            list_to_append_layer_to.extend([f'self.linear_{self.linear_count} = Dense({layer[1]}, '
                                            f'activation = "{activation}", kernel_initializer = {initialisation})'])
            tmp = f'self.linear_{self.linear_count}(x)'
            if self.batch_norm:
                tmp = f'self.batch_norm({tmp}, training = training)'
            if self.dropout != 0:
                tmp = f'self.dropout({tmp})'
            self.sum_up.extend([f'x = {tmp}'])
            self.linear_count += 1
        else:
            raise ValueError("Wrong layer name")
        self.last_layer_type = layer_name

    def create_batch_norm_layers(self):
        batch_norm_layers = []
        if self.batch_norm != 0:
            batch_norm_layers.extend([f'self.batch_norm = tf.keras.layers.BatchNormalization()'])
        return batch_norm_layers