from organ.ORGAN_0 import ORGAN
import tensorflow as tf
import os
import organ.mol_metrics as mm

DEFAULT_CONFIG = {'name' : 'test', 'params' : {'PRETRAIN_DIS_EPOCHS': 2, 'PRETRAIN_GEN_EPOCHS': 2,
                                               'GEN_ITERATIONS' : 1, 'DIS_EPOCHS' : 1, 'LAMBDA' : 0.5,
                                               'GEN_BATCH_SIZE' : 32, 'DIS_BATCH_SIZE' : 32}}

class Organ:
    def __init__(self, training_data, max_smiles=None, config_dict=DEFAULT_CONFIG):

        """
        Constructor wrapping that of the source ORGAN code. The config dict should contain the parameters
        listed in the original model. Valid parameters include in the following:



        - name. String which will be used to identify the
        model in any folders or files created.

        - input_data: csv file listing the training smiles with "smiles" header.

        - max_smiles: Maximum number of smiles to use from the input data (for prototyping purposes) - by default
        None, which will use all of the data.

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
        self.checkpoint_dir  = config_dict.get("checkpoint_dir", "ckpt")
        self.config_dict = config_dict

        # Load and process data, if applicable. Finalize setup.
        input = self.__process_data(training_data)
        if max_smiles is not None:
            max_smiles += 1  # The first input row is just the header, so add one
        self.model.load_training_set(input, num_samples=max_smiles)

    def train(self, num_iterations=1):

        """
        Light training wrapper for the ORGAN code taken from the original paper.


        :param num_iterations: Number of training iterations to perform.
        :return: None
        """
        self.model.set_training_program(['novelty'], [num_iterations]) # TODO: Decide what to do here with objectives
        self.model.load_metrics()
        self.model.train(ckpt_dir=self.checkpoint_dir)



    def sample(self, num_samples=1):
        """
        Sample some smiles from the model.

        :param num_samples: Number of smiles to generate.
        :return: Generated smiles.
        """

        # The ORGAN model can only sample multiples of batch_size number of samples
        num_batches = num_samples // self.model.GEN_BATCH_SIZE
        if num_samples % self.model.GEN_BATCH_SIZE > 0:
            num_batches += 1

        samples = self.model.generate_samples(num_batches)[:num_samples]
        samples = [mm.decode(sample, self.model.ord_dict) for sample in samples]

        return samples

    def __process_data(self, input_data):
        """
        Optional pre-processing code. Currently does nothing and assumes input is standardized.

        :param input_data:
        :return: input_data
        """
        return input_data

    def save(self):
        """
        Save the current model to checkpoint_dir using tensorflow's Saver utilities.

        :return: None
        """
        model_saver = tf.train.Saver()

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        ckpt_file = os.path.join(
            self.checkpoint_dir, '{}.ckpt'.format(self.model.PREFIX))

        model_saver.save(self.model.sess, ckpt_file)

    def load_weights(self, load_path = None):
        """
        Loads a saved model from load_path and updates weights accordingly. If None, defaults to checkpoint dir/name.ckpt for this class.
        Note that the class should be initialized with the SAME hyper-parameters that were used for training.
        This function just loads the tuned model parameters.

        :param load_path (str): Path from which to load the model.
        :return: None
        """

        if load_path is None:
            load_path = os.path.join(self.checkpoint_dir, self.model.PREFIX + ".ckpt")

        self.model.load_prev_training(load_path)


if __name__ == '__main__':
    # Sample execution
    organ = Organ(training_data='data/zinc.csv', max_smiles=32, config_dict=DEFAULT_CONFIG)

    print("Training ORGAN....")
    organ.train()

    print("Training finished. Saving model weights to disk.")
    organ.save()

    print("Saving finished. Loading model from disk.")
    organ2 = Organ(training_data='data/zinc.csv', max_smiles=32, config_dict=DEFAULT_CONFIG)
    organ2.load_weights()
    samples = organ2.sample(num_samples=None)

    print("Model loaded. Example samples generated: ")
    print(samples)

