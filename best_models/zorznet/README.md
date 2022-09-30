# ZorzNet model

Este modelo fue entrenado con las 500hs de entrenamiento de LibriSpeech. Los resultados obtenidos fueron:
- LER de validación: 7.2%
- WER de validación: 21.7%

Por otra parte, se realizó una validación con aproximadamente 1500 muestras de TIMIT, donde se obtuvieron los siguientes resultados:
- LER de validación: 15.2%
- WER de validación: 43.4%

## Configuración de hiperparámetros

La configuración de los hiperparámetros de la red fue la siguiente:

```
project_data = ProjectData()

network_data = ZorzNetData()
network_data.model_path = 'drive/My Drive/Tesis/repo/' + project_data.ZORZNET_MODEL_PATH
network_data.checkpoint_path = 'drive/My Drive/Tesis/repo/' + project_data.ZORZNET_CHECKPOINT_PATH
network_data.tensorboard_path = 'drive/My Drive/Tesis/repo/' + project_data.ZORZNET_TENSORBOARD_PATH

network_data.num_classes = ClassicLabel.num_classes - 1
network_data.num_features = 494
network_data.noise_stddev = 0.1
network_data.num_reduce_by_half = 0

network_data.num_dense_layers_1 = 1
network_data.num_units_1 = [400] * network_data.num_dense_layers_1
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.batch_normalization_trainable_1 = True
network_data.keep_prob_1 = [0.6] * network_data.num_dense_layers_1
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.is_bidirectional = True
network_data.num_cell_units = [512] * 2
network_data.cell_activation = [tf.nn.tanh] * 2
network_data.keep_prob_rnn = None#[0.8]
network_data.rnn_batch_normalization = True

network_data.num_dense_layers_2 = 1
network_data.num_units_2 = [150]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.batch_normalization_trainable_2 = True
network_data.keep_prob_2 = [0.6]
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2

network_data.dense_regularizer = 0.5
network_data.rnn_regularizer = 0.0

network_data.beam_width = 0    # 0 -> greedy_decoder, >0 -> beam_search

network_data.learning_rate = 0.001
network_data.use_learning_rate_decay = True
network_data.learning_rate_decay_steps = 5000
network_data.learning_rate_decay = 0.98

network_data.clip_gradient = 5
network_data.optimizer = 'adam'      # 'rms', 'adam', 'momentum', 'sgd'
network_data.momentum = None
```