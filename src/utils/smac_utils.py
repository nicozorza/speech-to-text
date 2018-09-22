import threading
import tensorflow as tf


def wait_for_user_input_non_block(data):
    def get_user_input(user_input_ref):
        user_input_ref[0] = input("press enter to final optimization loop:")

    mythread = threading.Thread(target=get_user_input, args=(data,))
    mythread.daemon = True
    mythread.start()
    print("mythread.start")


def remove_if_exist(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
