from organ.ORGAN_0 import ORGAN

DEFAULT_CONFIG = {'name' : 'test', 'params' : {'PRETRAIN_DIS_EPOCHS': 2, 'PRETRAIN_GEN_EPOCHS' : 2}}

class Organ:
    def __init__(self, config_dict=DEFAULT_CONFIG):

        """
        Constructor wrapping that of the source ORGAN code. The config dict should contain the parameters
        listed in the original model. Valid parameters include in the following:



        - name. String which will be used to identify the
        model in any folders or files created.

        - metrics_module. String identifying the module containing
        the metrics.

        - params. Optional. Dictionary containing the parameters
        that the user wishes to specify.

        The valid parameters are as follows:

        'WGAN' (bool) : Whether or not to use a WGAN penalty
        'PRETRAIN_GEN_EPOCHS' (int) : Number of pre-training epochs to perform for the generator
        'PRETRAIN_DIS_EPOCHS' (int) : Number of pre-training epochs to perform for the generator
        'GEN_ITERATIONS' (int) : Training iterations for generator within outer training loop
        'GEN_BATCH_SIZE' (int) : Batch size for generator network
        'SEED' (int) : random seed to be used
        'DIS_BATCH_SIZE' (int) : Batch size for discriminator network
        'DIS_EPOCHS' (int) : Number of epochs to perform for discriminator within outer loop
        'SAMPLE_NUM' (int) : Number of samples to generate during training
        'LAMBDA' : (float) : controls tradeoff between objective and generation

        There are other parameters, which are likely of lesser importance, see init code.

        - verbose. Boolean specifying whether output must be
        produced in-line.

        :param config_dict: Dictionary specifying the parameters to be used to initialize ORGAN.
        """
        self.model = ORGAN(**config_dict)
        self.config_dict = config_dict

    def train(self, input_data, num_iterations=1):

        # Load and process data, if applicable
        input = self.__process_data(input_data)
        self.model.load_training_set(input, num_samples=None)


        self.model.set_training_program(['novelty'], [num_iterations]) # TODO: Decide what to do here with objectives
        self.model.load_metrics()
        self.model.train(ckpt_dir=self.config_dict.get('checkpoint_dir', 'ckpt'))


    def __process_data(self, input_data):
        return input_data

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass


if __name__ == '__main__':
    #Debug
    organ = Organ()
    organ.train('data/zinc.csv')

