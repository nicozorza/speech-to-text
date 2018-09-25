from tensorboard import summary
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes, importer
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.summary import summary

input_graph = "out/model/model"
input_checkpoint = "out/checkpoint/"
output_graph = "out/mobile/frozen_model.pb"
output_graph_opt = "out/mobile/frozen_opt_model.pb"
input_node_names = ["input_features/input", "seq_len/sequence_length"]
output_node_names = "decoder/output_seq"

freeze_graph.freeze_graph(input_graph=input_graph,
                          input_saver="",
                          input_binary=True,
                          input_checkpoint=input_checkpoint,
                          output_node_names=output_node_names,
                          restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_graph,
                          clear_devices=True,
                          initializer_nodes="",
                          variable_names_whitelist="",
                          variable_names_blacklist="",
                          input_meta_graph="",
                          input_saved_model_dir="",
                          saved_model_tags=tag_constants.SERVING)


input_graph_def = graph_pb2.GraphDef()
with gfile.Open(output_graph, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    input_node_names,
    output_node_names.split(","), dtypes.float32.as_datatype_enum)
f = gfile.FastGFile(output_graph_opt, "w")
f.write(output_graph_def.SerializeToString())
f.close()

# def import_to_tensorboard(model_dir, log_dir):
#     with session.Session(graph=ops.Graph()) as sess:
#         with gfile.FastGFile(model_dir, "rb") as f:
#             graph_def = graph_pb2.GraphDef()
#             graph_def.ParseFromString(f.read())
#             importer.import_graph_def(graph_def)
#
#         pb_visual_writer = summary.FileWriter(log_dir)
#         pb_visual_writer.add_graph(sess.graph)
#         print("Model Imported. Visualize by running: "
#           "tensorboard --logdir={}".format(log_dir))
#
#
# import_to_tensorboard(output_graph_opt, "tl_test_board_mobile")
