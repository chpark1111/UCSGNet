'''
Code for reading initial configuration 
i.e. hyperparameters
'''
from configobj import ConfigObj
import os
import string

class Config(object):
    def __init__(self, filename: string):
        """
        Read from a config file
        :param filename: name of the file to read from
        """
        BASE = os.path.dirname(os.path.abspath(__file__))

        self.filename = filename
        config = ConfigObj(os.path.join(BASE, '../config/%s'%(self.filename)))

        self.config = config

        # Whether to turn on/off debug(does not make logs when on)
        self.debug = config.as_bool("debug")

        self.comment = config["comment"]

        self.model_path = config["train"]["model_name"]

        self.preload_model = config["train"].as_bool("preload_model")

        self.pretrain_modelpath = config["train"]["pretrain_model_path"]

        self.num_dim = config["train"].as_int("num_dim")

        self.epochs = config["train"].as_int("num_epochs")

        self.batch_size = config["train"].as_int("batch_size")

        self.latent_sz = config["train"].as_int("latent_sz")

        self.lr = config["train"].as_float("lr")

        self.optim = config["train"]["optim"]

        self.num_shape_type = config["train"].as_int("num_shape_type")

        self.beta1 = config["train"].as_float("beta1")
        self.beta2 = config["train"].as_float("beta2")

        self.points_per_sample_in_batch = config["train"].as_int("points_per_sample_in_batch")

        self.out_shape_layer = config["train"].as_int("out_shape_layer")

        self.threshold = config["train"].as_float("threshold")

        self.use_planes = config["train"].as_bool("use_planes")

        self.num_csg_layer = config["train"].as_int("num_csg_layer")

        self.worker = config["train"].as_int("worker")

        self.path = config["train"]["path"]

    def write_config(self, filename):
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        for attr, value in self.__dict__.items():
            print(attr, value)

if __name__ == "__main__":
    file = Config("config_2d.yml")
    file.write_config('test')