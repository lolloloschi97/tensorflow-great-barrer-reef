from tensorboard import program
from hyper_param import *
def launch_tensorboard(name_model):
    if WITH_COLAB == False:
        tracking_address = "../src/logs/" + name_model # the path of your log file.
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tracking_address])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")