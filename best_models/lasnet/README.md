# LASNet model

Este modelo fue entrenado con las 500hs de entrenamiento de LibriSpeech. Los resultados obtenidos fueron:
- LER de validación: 7.2%
- WER de validación: 17.9%

Por otra parte, se realizó una validación con aproximadamente 1500 muestras de TIMIT, donde se obtuvieron los siguientes resultados:
- LER de validación: 17.1%
- WER de validación: 43.9%

## Configuración de hiperparámetros

La configuración de los hiperparámetros de la red fue la siguiente:

```
project_data = ProjectData()

network_data = LASNetData()
network_data.model_path = 'drive/My Drive/Tesis/repo/' + project_data.LAS_NET_MODEL_PATH
network_data.checkpoint_path = 'drive/My Drive/Tesis/repo/' + project_data.LAS_NET_CHECKPOINT_PATH
network_data.tensorboard_path = 'drive/My Drive/Tesis/repo/' + project_data.LAS_NET_TENSORBOARD_PATH

network_data.num_classes = LASLabel.num_classes
network_data.num_features = 494
network_data.num_embeddings = 0
network_data.sos_id = LASLabel.SOS_INDEX
network_data.eos_id = LASLabel.EOS_INDEX
network_data.noise_stddev = 0.1
network_data.num_reduce_by_half = 0

network_data.beam_width = 0

network_data.num_dense_layers_1 = 2
network_data.num_units_1 = [400] * network_data.num_dense_layers_1
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.batch_normalization_trainable_1 = True
network_data.keep_prob_1 = [0.8] * network_data.num_dense_layers_1
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.listener_num_layers = 1
network_data.listener_num_units = [512] * network_data.listener_num_layers
network_data.listener_activation_list = [tf.nn.tanh] * network_data.listener_num_layers
network_data.listener_keep_prob_list = [0.9] * network_data.listener_num_layers

network_data.num_dense_layers_2 = 0
network_data.num_units_2 = [400]
network_data.dense_activations_2 = [tf.nn.tanh] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.batch_normalization_trainable_2 = True
network_data.keep_prob_2 = [0.8] * network_data.num_dense_layers_2
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2

network_data.attention_type = 'luong'       # 'luong', 'bahdanau'
network_data.attention_num_layers = 1
network_data.attention_size = None
network_data.attention_units = 512
network_data.attention_activation = tf.nn.tanh
network_data.attention_keep_prob = 0.9

network_data.kernel_regularizer = 0.0
network_data.sampling_probability = 0.1

network_data.learning_rate = 0.001
network_data.use_learning_rate_decay = True
network_data.learning_rate_decay_steps = 4000
network_data.learning_rate_decay = 0.98

network_data.clip_gradient = 0
network_data.optimizer = 'adam'      # 'rms', 'adam', 'momentum', 'sgd'
network_data.momentum = None
```
