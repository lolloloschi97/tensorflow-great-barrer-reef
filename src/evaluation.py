from tensorboard import program
def launch_tensorboard(name_model):
    tracking_address = "../src/logs/" + name_model # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")