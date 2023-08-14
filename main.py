import PySimpleGUI as sg
from nn_modeller.pytorch.NN import NN as PY
from nn_modeller.tensorflow.NN import NN as TFnn
from nn_modeller.tensorflow.CNN import CNN as TFcnn

sg.theme('LightGrey1')
sg.set_options(font = 'Arial 12', )
column_center = [  [sg.Button('Новый проект', key = '-btn_new_project')]
                   ]
str = 'Данное приложение реализует генерацию модели нейронной\nсети в соответствии с заданными параметрами.\n'
layout = [
    [sg.Text(str)],
    [sg.Push(), sg.Column(column_center,element_justification='c'), sg.Push()]
]

def create_options_window():
    column_center = [[sg.Button('Создать проект', key='-btn_create_project')]]
    layout = [
        [sg.Text('Выбор библиотеки реализации')],
        [sg.Radio('TensorFlow', "RADIO1", default=True, key="-r1", enable_events=True),
        sg.Radio('Pytorch', "RADIO1", default=False, key="-r2", enable_events=True)
    ],
        [sg.Text('Выбор типа модели')],
        [sg.Radio('Линейная', "RADIO2", default=True, key="-rl"),
         sg.Radio('Сверточная', "RADIO2", default=False, key="-rc")],
        [sg.Push(), sg.Column(column_center, element_justification='c'), sg.Push()]
    ]
    option_window = sg.Window("Настройка параметров модели", layout, modal=True, size=(400, 200))
    while True:
        event, value = option_window.read()
        if event == '-r2':
            option_window['-rc'].update(visible=False)
        if event == '-r1':
            option_window['-rc'].update(visible=True)
        if event == '-btn_create_project':
            if value['-r1'] == True:
                type = 'cnn'
                if value['-rl'] == True:
                    type = 'nn'
                create_tf_modal_window(type)
            else:
                create_torch_modal_window()
        if event == sg.WINDOW_CLOSED:
            break
    option_window.close()

def create_row_tf_nn(row_counter, row_number_view):
    row = []
    options_activation = ["elu", "exponential", "hard_sigmoid", "linear", "relu", "selu",
                          "sigmoid", "softmax", "softplus", "softsign", "tanh", "none"]
    options_initialization = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform",
                              "lecun_normal", "lecun_uniform", "truncated_normal",
                              "variance_scaling", "default"]
    row = [
        sg.pin(
            sg.Col([
                [sg.Text('X', enable_events=True, border_width=0, key=('-btn_del_elem', row_counter)),
                 sg.Input('units', size=(10, 1), key=('-layer', row_counter)),
                 sg.Combo(options_activation, size= (20, 1), key=('-activation', row_counter)),
                 sg.Combo(options_initialization, size=(20, 1), key=('-initialisation', row_counter))
                 ]
            ],
                key=('-row', row_counter)
            )
        )
    ]
    return row

def create_row_tf_cnn(row_counter, row_number_view, r_type):
    row = []
    options_activation = ["elu", "exponential", "hard_sigmoid", "linear", "relu", "selu",
                          "sigmoid", "softmax", "softplus", "softsign", "tanh", "none"]
    options_initialization = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform",
                              "lecun_normal", "lecun_uniform", "truncated_normal",
                              "variance_scaling", "default"]
    if r_type == 'conv':
        row = [
            sg.pin(
                sg.Col([
                    [sg.Text('X', enable_events=True, border_width=0, key=('-btn_del_elem', row_counter)),
                     sg.Text(r_type, key=('-type', row_counter)),
                     sg.Input('channels', size = (5, 1), key = ('-channels', row_counter)),
                     sg.Input('kernel size', size=(5, 1), key=('-kernel_size', row_counter)),
                     sg.Input('stride', size=(5, 1), key=('-stride', row_counter)),
                     sg.Combo(['same', 'valid'], size=(5, 1), key=('-padding', row_counter)),
                     sg.Combo(options_activation, size=(20, 1), key=('-activation', row_counter)),
                     sg.Combo(options_initialization, size=(20, 1), key=('-initialisation', row_counter))
                     ]
                ],
                    key=('-row', row_counter)
                )
            )
        ]
    elif r_type == 'maxpool' or r_type == 'avgpool':
        row = [
            sg.pin(
                sg.Col([
                    [sg.Text('X', enable_events=True, border_width=0, key=('-btn_del_elem', row_counter)),
                     sg.Text(r_type, key=('-type', row_counter)),
                     sg.Input('kernel size', size=(5, 1), key=('-kernel_size', row_counter)),
                     sg.Input('stride', size=(5, 1), key=('-stride', row_counter)),
                     sg.Combo(['same', 'valid'], size=(5, 1), key=('-padding', row_counter)),
                     sg.Combo(options_activation, size=(20, 1), key=('-activation', row_counter), visible= False),
                     sg.Combo(options_initialization, size=(20, 1), key=('-initialisation', row_counter), visible= False)
                     ]
                ],
                    key=('-row', row_counter)
                )
            )
        ]
    elif r_type == 'linear':
        row = [
            sg.pin(
                sg.Col([
                    [
                        sg.Text('X', enable_events=True, border_width=0, key=('-btn_del_elem', row_counter)),
                        sg.Text(r_type, key=('-type', row_counter)),
                        sg.Input('units', size=(5, 1), key=('-units', row_counter)),
                        sg.Combo(options_activation, size=(20, 1), key=('-activation', row_counter)),
                        sg.Combo(options_initialization, size=(20, 1), key=('-initialisation', row_counter))
                    ]
                ],
                    key=('-row', row_counter)
                )
            )
        ]
    return row

def create_tf_modal_window(win_type):
    row = []
    add_layer = []
    descr = []
    row_counter = 0
    row_number_view = 1
    layers_type = []
    options_activation = ["elu", "exponential", "hard_sigmoid", "linear", "relu", "selu",
                          "sigmoid", "softmax", "softplus", "softsign", "tanh", "none"]
    options_initialization = ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform",
                              "lecun_normal", "lecun_uniform", "truncated_normal",
                              "variance_scaling", "default"]
    if win_type == 'nn':
        row = create_row_tf_nn(0, 1)
        add_layer = [sg.Text('+', enable_events= True, key = '-btn_add')]
        descr = [sg.Text('        Size'), sg.Text('              Activation'), sg.Text('                             Initialization')]
    else:
        #row = create_row_tf_cnn(0, 1, 'conv')
        row = []
        row_counter -= 1
        row_number_view -= 1
        add_layer = [sg.Text('+ conv', enable_events= True, key = '-btn_add_conv'),
                     sg.Text('+ maxpool', enable_events=True, key = '-btn_add_mp'),
                     sg.Text('+ avgpool', enable_events=True, key = '-btn_add_ap'),
                     sg.Text('+ linear', enable_events=True, key = '-btn_add_l')]

    desctiption_column = [
        [sg.Text('Tensorflow ')],
        [sg.Text('Activation'), sg.Combo(options_activation, size=(20, 1), key=('-activation_all'))],
        [sg.Text('Initialization'), sg.Combo(options_initialization, size=(20, 1), key=('-initialisation_all'))],
        descr,
        [sg.Column([row], key='-row_panel')],
        add_layer,
        [sg.Text('Пакетная нормализация'), sg.Combo(['True', 'False'], size=(10, 1), key='-batch_norm')],
        [sg.Text('Dropout'), sg.Input('0', size=(10, 1), key='-dropout')]
        #[sg.Text('+', enable_events= True, key = '-btn_add')]
    ]
    output_column = [
        [sg.Button('Собрать модель', key = '-btn_run'), sg.Button('Выход', key='-btn_exit')],
        [sg.Multiline(size = (40, 20),key='-ml_output', horizontal_scroll=True)]
    ]
    new_area_layot = [
        [
        sg.Column(desctiption_column),
        sg.VSeparator(),
        sg.Column(output_column)
        ]
    ]
    tf_modal_window = sg.Window('TensorFlow ' + win_type, new_area_layot, modal = True)


    rows_invisible = []
    event, values = tf_modal_window.read()

    while True:
        event, values = tf_modal_window.read()

        if event == '-btn_add':
            row_counter += 1
            row_number_view += 1
            if win_type == 'nn':
                tf_modal_window.extend_layout(tf_modal_window['-row_panel'],
                                              [create_row_tf_nn(row_counter, row_number_view)])
        if event == '-btn_add_conv':
            row_counter += 1
            row_number_view += 1
            layers_type.append('conv')
            tf_modal_window.extend_layout(tf_modal_window['-row_panel'],
                                          [create_row_tf_cnn(row_counter, row_number_view, 'conv')])
        if event == '-btn_add_mp':
            row_counter += 1
            row_number_view += 1
            layers_type.append('maxpool')
            tf_modal_window.extend_layout(tf_modal_window['-row_panel'],
                                          [create_row_tf_cnn(row_counter, row_number_view, 'maxpool')])
        if event == '-btn_add_ap':
            row_counter += 1
            row_number_view += 1
            layers_type.append('avgpool')
            tf_modal_window.extend_layout(tf_modal_window['-row_panel'],
                                          [create_row_tf_cnn(row_counter, row_number_view, 'avgpool')])
        if event == '-btn_add_l':
            row_counter += 1
            row_number_view += 1
            layers_type.append('linear')
            tf_modal_window.extend_layout(tf_modal_window['-row_panel'],
                                          [create_row_tf_cnn(row_counter, row_number_view, 'linear')])

        if event =='-btn_run':
            if win_type == 'nn':
                activation = ''
                init = ''
                output_activation = ''
                layers_info = []
                dropout = 0
                bn = True
                try:
                    if values['-activation_all'] != '':
                        activation = values['-activation_all']
                    else:
                        activation = []
                        for i in range(row_counter + 1):
                            if not (i in set(rows_invisible)):
                                if values[('-activation', i)] != '':
                                    activation.append(values[('-activation', i)])
                                else:
                                    activation.append('none')

                    i = row_counter
                    while i in set(rows_invisible):
                        i -= 1
                    output_activation = values[('-activation', i)]
                    if output_activation == '':
                        output_activation = 'none'
                    if values['-initialisation_all'] != '':
                        init = values['-initialisation_all']
                    else:
                        init = []
                        for i in range(row_counter + 1):
                            if not (i in set(rows_invisible)):
                                if values[('-initialisation', i)] != '':
                                    init.append(values[('-initialisation', i)])
                                else:
                                    init.append('default')

                    for i in range(row_counter + 1):
                        if not (i in set(rows_invisible)):
                            layers_info.append(int(values[('-layer', i)]))

                    dropout = float(values['-dropout'])

                    if values['-batch_norm'] == '' or values['-batch_norm'] == 'False':
                        bn = False

                    model = TFnn(layers_info=layers_info, output_activation=output_activation,
                                 hidden_activations=activation, dropout=dropout,
                                 initialiser=init, batch_norm=bn)
                    res = model.get_model_summary()
                    tf_modal_window['-ml_output'].Update(res)
                except:
                    tf_modal_window['-ml_output'].Update('Ошибка входных данных')
            elif win_type == 'cnn':
                activation = ''
                init = ''
                output_activation = ''
                layers_info = []
                dropout = 0
                bn = True
                try:
                    if values['-activation_all'] != '':
                        activation = values['-activation_all']
                    else:
                        activation = []
                        for i in range(row_counter + 1):
                            if not (i in set(rows_invisible)):
                                if values[('-activation', i)] != '':
                                    activation.append(values[('-activation', i)])
                                else:
                                    activation.append('none')

                    i = row_counter
                    while i in set(rows_invisible):
                        i -= 1
                    output_activation = values[('-activation', i)]
                    if output_activation == '':
                        output_activation = 'none'
                    if values['-initialisation_all'] != '':
                        init = values['-initialisation_all']
                    else:
                        init = []
                        for i in range(row_counter + 1):
                            if not (i in set(rows_invisible)):
                                if values[('-initialisation', i)] != '':
                                    init.append(values[('-initialisation', i)])
                                else:
                                    init.append('default')

                    for i in range(row_counter + 1):
                        if not (i in set(rows_invisible)):
                            if layers_type[i] == 'conv':
                                layers_info.append(
                                    ['conv', int(values[('-channels', i)]), int(values['-kernel_size', i]),
                                     int(values[('-stride', i)]), values[('-padding', i)]])
                            elif layers_type[i] == 'maxpool' or layers_type[i] == 'avgpool':
                                layers_info.append([layers_type[i], int(values['-kernel_size', i]),
                                                    int(values[('-stride', i)]), values[('-padding', i)]])
                            else:
                                layers_info.append(['linear', int(values[('-units', i)])])

                    dropout = float(values['-dropout'])

                    if values['-batch_norm'] == '' or values['-batch_norm'] == 'False':
                        bn = False

                    '''model = TFnn(layers_info=layers_info, output_activation=output_activation,
                                 hidden_activations=activation, dropout=dropout,
                                 initialiser=init, batch_norm=bn)'''
                    model = TFcnn(layers_info=layers_info, output_activation=output_activation,
                                  hidden_activations=activation, dropout=dropout,
                                  initialiser=init, batch_norm=bn)
                    res = model.get_model_summary()
                    tf_modal_window['-ml_output'].Update(res)
                except:
                    tf_modal_window['-ml_output'].Update('Ошибка входных данных')

        if event in (sg.WINDOW_CLOSED, '-btn_exit'):
            break

        elif event[0] == '-btn_del_elem':
            row_number_view -= 1
            rows_invisible.append(event[1])
            tf_modal_window[('-row', event[1])].update(visible = False)



    tf_modal_window.close()

def create_row_pt_nn (row_counter, row_number_view):
    row = []
    options_activation = ["elu", "hardshrink", "hardtanh", "leakyrelu", "logsigmoid",
                          "prelu", "relu", "relu6", "rrelu", "selu", "sigmoid", "softplus",
                          "logsoftmax", "softshrink", "softsign", "tanh", "tanhshrink",
                          "softmin", "softmax", "none"]
    options_initialization = ["uniform", "normal", "eye", "xavier_uniform", "xavier",
                              "xavier_normal", "kaiming_uniform", "kaiming", "kaiming_normal",
                              "he", "orthogonal", "default"]
    row = [
        sg.pin(
            sg.Col([
                [sg.Text('X', enable_events=True, border_width=0, key=('-btn_del_elem', row_counter)),
                 sg.Input('units', size=(10, 1), key=('-layer', row_counter)),
                 # sg.Input(size=(20, 1), key=('-activation', row_counter)),
                 sg.Combo(options_activation, size=(20, 1), key=('-activation', row_counter)),
                 # sg.Input(size=(20, 1), key=('-initialisation', row_counter))#,
                 sg.Combo(options_initialization, size=(20, 1), key=('-initialisation', row_counter))
                 # sg.Text(f'{row_number_view}', key=('-row_num', row_counter))
                 ]
            ],
                key=('-row', row_counter)
            )
        )
    ]
    return row

def create_torch_modal_window():
    options_activation = ["elu", "hardshrink", "hardtanh", "leakyrelu", "logsigmoid",
                          "prelu", "relu", "relu6", "rrelu", "selu", "sigmoid", "softplus",
                          "logsoftmax", "softshrink", "softsign", "tanh", "tanhshrink",
                          "softmin", "softmax", "none"]
    options_initialization = ["uniform", "normal", "eye", "xavier_uniform", "xavier",
                              "xavier_normal", "kaiming_uniform", "kaiming", "kaiming_normal",
                              "he", "orthogonal", "default"]
    row = create_row_pt_nn(0, 1)
    desctiption_column = [
        [sg.Text('Pytorch')],
        [sg.Text('Входной слой'), sg.Input('units', size=(10, 1), key=('-input_dim'))],
        [sg.Text('Activation'), sg.Combo(options_activation, size=(20, 1), key=('-activation_all'))],
        [sg.Text('Initialization'), sg.Combo(options_initialization, size=(20, 1), key=('-initialisation_all'))],
        [sg.Text('        Size'), sg.Text('              Activation'),
         sg.Text('                             Initialization')],
        [sg.Column([row], key='-row_panel')],
        [sg.Text('+', enable_events=True, key='-btn_add')],
        [sg.Text('Пакетная нормализация'), sg.Combo(['True', 'False'], size=(10, 1), key='-batch_norm')],
        [sg.Text('Dropout'), sg.Input('0', size=(10, 1), key='-dropout')]
    ]
    output_column = [
        [sg.Button('Собрать модель', key = '-btn_run'), sg.Button('Выход', key='-btn_exit')],
        [sg.Multiline(size = (40, 20), key='-ml_output', horizontal_scroll=True)]
    ]
    new_area_layout = [
        [
            sg.Column(desctiption_column),
            sg.VSeparator(),
            sg.Column(output_column)
        ]
    ]

    torch_modal_window = sg.Window('Pytorch', new_area_layout, modal=True)
    row_counter = 0
    row_number_view = 1
    rows_invisible = []
    event, values = torch_modal_window.read()

    while True:
        event, values =torch_modal_window.read()

        if event == '-btn_add':
            row_counter += 1
            row_number_view += 1
            torch_modal_window.extend_layout(torch_modal_window['-row_panel'],
                                          [create_row_pt_nn(row_counter, row_number_view)])

        if event == '-btn_run':
            input_dim = 0
            activation = ''
            init = ''
            output_activation = ''
            layers_info = []
            dropout = 0
            bn = True
            try:
                input_dim = int(values['-input_dim'])

                if values['-activation_all'] != '':
                    activation = values['-activation_all']
                else:
                    activation = []
                    for i in range(row_counter + 1):
                        if not (i in set(rows_invisible)):
                            if values[('-activation', i)] != '':
                                activation.append(values[('-activation', i)])
                            else:
                                activation.append('none')

                i = row_counter
                while i in set(rows_invisible):
                    i -= 1
                output_activation = values[('-activation', i)]
                if output_activation == '':
                    output_activation = 'none'
                if values['-initialisation_all'] != '':
                    init = values['-initialisation_all']
                else:
                    init = []
                    for i in range(row_counter + 1) :
                        if not (i in set(rows_invisible)):
                            if values[('-initialisation', i)] != '':
                                init.append(values[('-initialisation', i)])
                            else:
                                init.append('default')

                for i in range(row_counter + 1):
                    if not (i in set(rows_invisible)):
                        layers_info.append(int(values[('-layer', i)]))

                dropout = float(values['-dropout'])

                if values['-batch_norm'] == '' or values['-batch_norm'] == 'False':
                    bn = False
                if input_dim <= 0:
                    torch_modal_window['-ml_output'].Update('Входной слой должен содержать целое число нейронов')
                    continue
                print(init)
                model = PY(input_dim= input_dim, layers_info=layers_info,
                          output_activation=output_activation, hidden_activations=activation,
                          dropout=dropout, initialiser=init, batch_norm=bn)
                res = model.get_model_summary()
                print(res)
                torch_modal_window['-ml_output'].Update(res)
            except:
                #torch_modal_window['-ml_output'].Update(init)
                torch_modal_window['-ml_output'].Update('Ошибка входных данных')

        if event in (sg.WINDOW_CLOSED, '-btn_exit'):
            break
        elif event[0] == '-btn_del_elem':
            row_number_view -= 1
            rows_invisible.append(event[1])
            torch_modal_window[('-row', event[1])].update(visible = False)

    torch_modal_window.close()

window = sg.Window('Modeller', layout, size=(500, 170))

while True:
    event, values = window.read()

    if event == '-btn_new_project':
        create_options_window()

    if event == sg.WINDOW_CLOSED:
        break

window.close()