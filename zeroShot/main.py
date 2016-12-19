from __future__ import division
from __future__ import print_function

from model import AttentionNN
from data import read_vocabulary
from test_iterator import test_iterator

import random
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer("max_size", 30, "Maximum sentence length [30]")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 123, "Value of random seed [123]")
flags.DEFINE_integer("epochs", 10, "Number of epochs to run [10]")
flags.DEFINE_integer("emb_size", 128, "Dimensons of embedding vector [128]")
flags.DEFINE_integer("hidden_size", 512, "Size of hidden units [512]")
flags.DEFINE_integer("num_layers", 4, "Depth of RNNs [4]")
flags.DEFINE_integer("patience", 100, "Number of steps before updating learning rate [100]")
flags.DEFINE_float("dropout", 0.2, "Dropout probability [0.0]")
flags.DEFINE_float("minval", -0.1, "Minimum value for initialization [-0.1]")
flags.DEFINE_float("maxval", 0.1, "Maximum value for initialization [0.1]")
flags.DEFINE_float("lr_init", 1.0, "Initial learning rate [1.0]")
flags.DEFINE_float("max_grad_norm", 5.0, "Maximum gradient cutoff [5.0]")
flags.DEFINE_string("optimizer_name", "SGD", "Optimizer choice [SGD]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory [checkpoints]")
flags.DEFINE_string("dataset", "debug", "Dataset to use [debug]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for training [False]")
flags.DEFINE_boolean("show", False, "Show progress [True]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


class debug:
    source_data_path      = "data/debug/train.en"
    target_data_path      = "data/debug/train.vi"
    source_vocab_path     = "data/debug/vocab.en"
    target_vocab_path     = "data/debug/vocab.vi"
    test_source_data_path = "data/debug/train.en"
    test_target_data_path = "data/debug/train.vi"
    prediction_data_path  = "data/debug/prediction.vi" 
 
class small:
    source_data_path      = "data/small/train.en"
    target_data_path      = "data/small/train.vi"
    source_vocab_path     = "data/small/vocab.en"
    target_vocab_path     = "data/small/vocab.vi"
    test_source_data_path = "data/small/tst2012.en"
    test_target_data_path = "data/small/tst2012.vi"
    prediction_data_path  = "data/small/prediction.vi" 



class medium:
    source_data_path      = "data/medium/train.en"
    target_data_path      = "data/medium/train.de"
    source_vocab_path     = "data/medium/vocab.en"
    target_vocab_path     = "data/medium/vocab.de"
    test_source_data_path = "data/medium/test.en"
    test_target_data_path = "data/medium/test.de"


def main(_):
    config = FLAGS
    if config.dataset == "small":
        data_config = small
    elif config.dataset == "medium":
        data_config = medium
    elif config.dataset == "default":
        data_config = default 
    elif config.dataset == "debug":
        data_config = debug
    else:
        raise Exception("[!] Unknown dataset {}".format(config.dataset))

    config.source_data_path      = data_config.source_data_path
    config.target_data_path      = data_config.target_data_path
    config.source_vocab_path     = data_config.source_vocab_path
    config.target_vocab_path     = data_config.target_vocab_path
    config.test_source_data_path = data_config.test_source_data_path 
    config.test_target_data_path = data_config.test_target_data_path 
    config.prediction_data_path  = data_config.prediction_data_path
    
    s_nwords  = len(read_vocabulary(config.source_vocab_path))
    t_nwords  = len(read_vocabulary(config.target_vocab_path))

    config.s_nwords  = s_nwords
    config.t_nwords  = t_nwords
    with tf.Session() as sess:
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        if not config.is_test:
            print ("==========================================================")
            print ("==================== Training Mode =======================")
            print ("==========================================================")
            print ("-------------------- Test iterator... --------------------")
            test_iterator(config.source_data_path, 
                    config.target_data_path, 
                    read_vocabulary(config.source_vocab_path),
                    read_vocabulary(config.target_vocab_path),
                    config.max_size,
                    config.batch_size)
            print ("-------------------- Start building model... -------------")
            attn = AttentionNN(config, sess)
            print ("") 
            print ("-------------------- Start training... -------------------")
            attn.train()
        else:
            print ("==========================================================")
            print ("===================== Testing Mode =======================")
            print ("==========================================================")
            attn = AttentionNN(config, sess)
            print ("--------------------- Start testing... -------------------")
            attn.sample() 
            #perplexity = attn.test()
            #print("Perplexity: {}".format(perplexity))


if __name__ == "__main__":
    tf.app.run()
