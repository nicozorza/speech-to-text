import os
import queue
import shutil
import time
from threading import Thread

import tensorflow as tf
import numpy as np
import logging

from ConfigSpace.conditions import InCondition
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from tensorflow.contrib.layers import l2_regularizer

from src.neural_network.data_conversion import indexToStr
from src.neural_network.NetworkData import NetworkData
from src.neural_network.RNN import RNNClass
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData
from src.utils.smac_utils import wait_for_user_input_non_block, remove_if_exist


project_data = ProjectData()
train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
val_database = Database.fromFile(project_data.VAL_DATABASE_FILE, project_data)

# TODO Add a different method for this
train_feats, train_labels, _, _, _, _ = train_database.split_sets(1.0, 0.0, 0.0)
val_feats, val_labels, _, _, _, _ = val_database.split_sets(1.0, 0.0, 0.0)

# -----------------------------------------------------------------------------------------
optimization_epochs = 100
validation_epochs = 200
# space_optimization_evals = 1
batch_size = 50

run_folder = 'run'

run_count_step = 1

restore_prev_run = False
restore_prev_run_folder = 'ctc_network/out/smac/run_1/'
results_dir = 'ctc_network/out/smac/{}/'.format(run_folder)
# -----------------------------------------------------------------------------------------


space = {
    'input_dense_depth': CategoricalHyperparameter("input_dense_depth", ["1", "2"], default_value="1"),
    'input_dense_1': UniformIntegerHyperparameter("input_dense_1", 50, 250, default_value=100),
    'input_dense_2': UniformIntegerHyperparameter("input_dense_2", 50, 250, default_value=100),

    'out_dense_depth': CategoricalHyperparameter("out_dense_depth", ["1", "2"], default_value="1"),
    'out_dense_1': UniformIntegerHyperparameter("out_dense_1", 50, 250, default_value=100),
    'out_dense_2': UniformIntegerHyperparameter("out_dense_2", 50, 250, default_value=100),

    'rnn_depth': CategoricalHyperparameter("rnn_depth", ["1", "2"], default_value="1"),
    'fw_1': UniformIntegerHyperparameter("fw_1", 10, 250, default_value=100),
    'fw_2': UniformIntegerHyperparameter("fw_2", 10, 250, default_value=100),
    'bw_1': UniformIntegerHyperparameter("bw_1", 10, 250, default_value=100),
    'bw_2': UniformIntegerHyperparameter("bw_2", 10, 250, default_value=100),

    'dense_regularizer': UniformFloatHyperparameter("dense_regularizer", 0, 1, default_value=0.0),
    'rnn_regularizer': UniformFloatHyperparameter("rnn_regularizer", 0, 1, default_value=0.0),

    #'use_dropout': CategoricalHyperparameter("use_dropout", [True, False], default_value=True),
    'input_keep_1': UniformFloatHyperparameter("input_keep_1", 0.5, 1, default_value=1),
    'input_keep_2': UniformFloatHyperparameter("input_keep_2", 0.5, 1, default_value=1),
    'output_keep_1': UniformFloatHyperparameter("output_keep_1", 0.5, 1, default_value=1),
    'output_keep_2': UniformFloatHyperparameter("output_keep_2", 0.5, 1, default_value=1),


}


hyper_space_conditions = [
    InCondition(child=space['input_dense_2'], parent=space['input_dense_depth'], values=["2"]),
    InCondition(child=space['out_dense_2'], parent=space['out_dense_depth'], values=["2"]),
    InCondition(child=space['fw_2'], parent=space['rnn_depth'], values=["2"]),
    InCondition(child=space['bw_2'], parent=space['rnn_depth'], values=["2"]),
    InCondition(child=space['input_keep_2'], parent=space['input_dense_depth'], values=["2"]),
    InCondition(child=space['output_keep_2'], parent=space['out_dense_depth'], values=["2"]),
]


def get_network_data(args):

    print(args)

    network_data = NetworkData()
    network_data.model_path = project_data.MODEL_PATH
    network_data.checkpoint_path = project_data.CHECKPOINT_PATH
    network_data.tensorboard_path = project_data.TENSORBOARD_PATH

    network_data.num_classes = ord('z') - ord('a') + 1 + 1 + 1
    network_data.num_features = 26

    network_data.num_input_dense_layers = int(args['input_dense_depth'])
    if network_data.num_input_dense_layers == 1:
        network_data.num_input_dense_units = [args['input_dense_1']]
    else:
        network_data.num_input_dense_units = [args['input_dense_1'], args['input_dense_2']]

    network_data.input_dense_activations = [tf.nn.tanh] * network_data.num_input_dense_layers
    network_data.input_batch_normalization = True

    network_data.is_bidirectional = True
    network_data.rnn_regularizer = args['rnn_regularizer']
    if int(args['rnn_depth']) == 1:
        network_data.num_fw_cell_units = [args['fw_1']]
        network_data.num_bw_cell_units = [args['bw_1']]
    else:
        network_data.num_fw_cell_units = [args['fw_1'], args['fw_2']]
        network_data.num_bw_cell_units = [args['bw_1'], args['bw_2']]
    network_data.cell_fw_activation = [tf.nn.tanh] * len(network_data.num_fw_cell_units)
    network_data.cell_bw_activation = [tf.nn.tanh] * len(network_data.num_bw_cell_units)

    network_data.num_dense_layers = int(args['out_dense_depth'])
    if network_data.num_dense_layers == 1:
        network_data.num_dense_units = [args['out_dense_1']]
    else:
        network_data.num_dense_units = [args['out_dense_1'], args['out_dense_2']]
    network_data.dense_activations = [tf.nn.tanh] * network_data.num_dense_layers
    network_data.dense_regularizer = args['dense_regularizer']
    network_data.dense_batch_normalization = True

    network_data.out_activation = None
    network_data.out_regularizer_beta = 0.0
    network_data.out_regularizer = l2_regularizer(network_data.out_regularizer_beta)

    network_data.learning_rate = 0.001
    network_data.adam_epsilon = 0.005
    network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                    epsilon=network_data.adam_epsilon)

    network_data.use_dropout = True # args['use_dropout']
    if network_data.num_input_dense_layers == 1:
        network_data.keep_dropout_input = [args['input_keep_1']]
    else:
        network_data.keep_dropout_input = [args['input_keep_1'], args['input_keep_2']]
    if network_data.num_dense_layers == 1:
        network_data.keep_dropout_output = [args['output_keep_1']]
    else:
        network_data.keep_dropout_output = [args['output_keep_1'], args['output_keep_2']]

    network_data.decoder_function = tf.nn.ctc_greedy_decoder

    return network_data


def objective(args):
    network_data = get_network_data(args)

    network = RNNClass(network_data)
    network.create_graph()

    network.train(
        train_features=train_feats,
        train_labels=train_labels,
        restore_run=False,
        save_partial=False,
        # save_freq=10,
        use_tensorboard=False,
        # tensorboard_freq=10,
        training_epochs=optimization_epochs,
        batch_size=batch_size
    )

    ler, loss = network.validate(val_feats, val_labels, show_partial=False, batch_size=batch_size)

    return ler


remove_if_exist(results_dir)

logger = logging.getLogger("Hyperparameter optimization")
logging.basicConfig(level=logging.INFO)

config_space = ConfigurationSpace()
config_space.add_hyperparameters(list(space.values()))
config_space.add_conditions(hyper_space_conditions)


scenario_dict = {"run_obj": "quality",
                 "runcount-limit": run_count_step,
                 "cs": config_space,
                 "deterministic": "true",
                 "output-dir": results_dir + 'log_0/'
                 }

scenario = Scenario(scenario_dict)
runhistory = RunHistory(aggregate_func=None)
stats = Stats(scenario)

if restore_prev_run:
    prev_results_dir = restore_prev_run_folder

    rh_path = os.path.join(prev_results_dir, "runhistory.json")
    runhistory.load_json(rh_path, scenario.cs)

    stats_path = os.path.join(prev_results_dir, "stats.json")
    stats.load(stats_path)

    traj_path = os.path.join(prev_results_dir, "traj_aclib2.json")
    trajectory = TrajLogger.read_traj_aclib_format(
        fn=traj_path, cs=scenario.cs)
    incumbent = trajectory[-1]["incumbent"]

    # new_traj_dir = '{}/log/run_1/'.format(results_dir)
    # new_traj_path = os.path.join(new_traj_dir, "traj_aclib2.json")
    # os.makedirs(new_traj_dir)
    # shutil.copy(traj_path, new_traj_path)

    smac = SMAC(scenario=scenario,
                runhistory=runhistory,
                stats=stats,
                restore_incumbent=incumbent,
                rng=np.random.RandomState(42),
                tae_runner=objective)
else:

    smac = SMAC(scenario=scenario,
                runhistory=runhistory,
                stats=stats,
                rng=np.random.RandomState(42),
                tae_runner=objective)

user_cmd = [None]
wait_for_user_input_non_block(user_cmd)
print("wait_for_user_input_non_block")

optimized_hyper = smac.optimize()

print("optimize")
iteration_index = 0
while user_cmd[0] is None:
    iteration_index += 1
    rh_path = os.path.join(scenario_dict["output-dir"] + "run_1/", "runhistory.json")
    stats_path = os.path.join(scenario_dict["output-dir"] + "run_1/", "stats.json")

    scenario_dict["runcount-limit"] += run_count_step
    scenario_dict["output-dir"] = results_dir + 'log_{}/'.format(iteration_index)
    scenario = Scenario(scenario_dict)

    runhistory = RunHistory(aggregate_func=None)
    runhistory.load_json(rh_path, scenario.cs)

    stats = Stats(scenario)
    stats.load(stats_path)

    smac = SMAC(scenario=scenario,
                runhistory=runhistory,
                stats=stats,
                restore_incumbent=optimized_hyper,
                rng=np.random.RandomState(42),
                tae_runner=objective)

    optimized_hyper = smac.optimize()


print("optimized hyperparameters:")
print(optimized_hyper)

optimized_net_data = get_network_data(optimized_hyper)
optimized_net = RNNClass(optimized_net_data)
optimized_net.create_graph()

optimized_net.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=False,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=10,
    training_epochs=validation_epochs,
    batch_size=batch_size
)


optimized_net.validate(val_feats, val_labels, show_partial=False)

print('Predicted: {}'.format(optimized_net.predict(val_feats[0])))
print('Target: {}'.format(indexToStr(val_labels[0])))

with open(results_dir+'optimized_hyperparameters.txt', 'w') as file:
    for key, value in optimized_hyper.get_dictionary().items():
        file.write('%s:%s\n' % (key, value))
