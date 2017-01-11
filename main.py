from __future__ import division
from __future__ import print_function

import os, sys, inspect, logging
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from model.attention import AttentionNN
from data.iterator import * 
from bleu.length_analysis import process_files       

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
flags.DEFINE_string("checkpointName", None, "Checkpoint name [None]")
flags.DEFINE_string("dataset", "small", "Dataset to use [small]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for training [False]")
flags.DEFINE_boolean("reload", False, "Reload previous experiments [False]")
flags.DEFINE_boolean("show", False, "Show progress [True]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

class debug:
    train_source_data_path = "data/debug/train.en"
    train_target_data_path = "data/debug/train.vi"
    source_vocab_path      = "data/debug/vocab.en"
    target_vocab_path      = "data/debug/vocab.vi"
    test_source_data_path  = "data/debug/test.en"
    test_target_data_path  = "data/debug/test.vi"
    valid_source_data_path = "data/debug/valid.en"
    valid_target_data_path = "data/debug/valid.vi"
 
class small:
    train_source_data_path = "data/small/train.en"
    train_target_data_path = "data/small/train.vi"
    source_vocab_path      = "data/small/vocab.en"
    target_vocab_path      = "data/small/vocab.vi"
    test_source_data_path  = "data/small/test.en"
    test_target_data_path  = "data/small/test.vi"
    valid_source_data_path = "data/small/valid.en"
    valid_target_data_path = "data/small/valid.vi"

class medium:
    train_source_data_path = "data/medium/train.en"
    train_target_data_path = "data/medium/train.vi"
    source_vocab_path      = "data/medium/vocab.en"
    target_vocab_path      = "data/medium/vocab.vi"
    test_source_data_path  = "data/medium/test.en"
    test_target_data_path  = "data/medium/test.vi"
    valid_source_data_path = "data/medium/valid.en"
    valid_target_data_path = "data/medium/valid.vi"

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
    
    iterator = DataIter(data_config.train_source_data_path, 
                        data_config.train_target_data_path,
                        data_config.source_vocab_path,
                        data_config.target_vocab_path,
                        config.max_size,
                        config.batch_size,
                        data_config.valid_source_data_path, 
                        data_config.valid_target_data_path,
                        data_config.test_source_data_path, 
                        data_config.test_target_data_path
                        )
   
    config.source_data_path      = data_config.train_source_data_path
    config.target_data_path      = data_config.train_target_data_path
    config.source_vocab_path     = data_config.source_vocab_path
    config.target_vocab_path     = data_config.target_vocab_path
    config.test_source_data_path = data_config.test_source_data_path 
    config.test_target_data_path = data_config.test_target_data_path 
    
    config.iterator  = iterator
    config.s_nwords  = iterator.source_vocab_size
    config.t_nwords  = iterator.target_vocab_size

    tf_config = tf.ConfigProto()
    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #tf_config.gpu_options.allocator_type = 'BFC' 
    #tf_config.gpu_options.allow_growth = False
    with tf.Session(config = tf_config) as sess:
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        if not config.is_test:
            print ("==========================================================")
            print ("==================== Training Mode =======================")
            print ("==========================================================")
            print ("-------------------- Test iterator... --------------------")
            iterator.test() 
            print ("-------------------- Start building model... -------------")
            attn = AttentionNN(config, sess)
            if config.reload:
                attn.load() 
            print ("") 
            print ("-------------------- Start training... -------------------")
            attn.train()
        else:
            print ("==========================================================")
            print ("===================== Testing Mode =======================")
            print ("==========================================================")
            attn = AttentionNN(config, sess)
            attn.load()
            print ("--------------------- Start testing... -------------------")
            attn.sample() 
            #perplexity = attn.test()
            #print("Perplexity: {}".format(perplexity))

if __name__ == "__main__":
    logging.basicConfig(filename='log.train', level=logging.DEBUG) 
    tf.app.run()
