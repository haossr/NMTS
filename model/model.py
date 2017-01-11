from __future__ import division
from __future__ import print_function

import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from datetime import datetime
from bleu.length_analysis import process_files
from utils import *

import tensorflow as tf
import numpy as np
import math
import os
import time
import collections

class Model(object):
    def timeit(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print ('[{}'.format(method.__name__), 'takes {0:.2f} sec]'.format(te-ts))
            return result
        return timed
   
    def get_log_name(self):
        date = datetime.now()
        return "{}-{}-{}-{}-{}-{}".format(self.name, self.dataset, date.month, date.day, date.hour, date.minute)
    
    def __init__(self, config, sess):
        raise NotImplementedError

    @timeit
    def build_variables(self): 
        raise NotImplementedError

    @timeit
    def build_graph(self):
        raise NotImplementedError

    @timeit
    def build_optimizer(self):
        if False: 
            self.learting_rate  = tf.train.exponential_decay(
                    learning_rate   = self.lr_init,
                    global_step     = self.global_step,
                    decay_steps     = 10000,
                    decay_rate      = 0.96,
                    staircase       = True)  
        self.optimizer = tf.contrib.layers.optimize_loss(
                        loss            = self.loss, 
                        global_step     = self.global_step,
                        learning_rate   = self.learning_rate, 
                        optimizer       = self.optimizer_name, 
                        clip_gradients  = self.max_grad_norm)

    @timeit
    def build_other_helpers(self):
        self.saver = tf.train.Saver(tf.trainable_variables())
        #tf.scalar_summary("loss", self.loss)
        #tf.scalar_summary("best training loss", self.best_loss)
        self.summarizer = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("./logs/{}".format(self.log_name().eval()),\
                 self.sess.graph)

    @timeit
    def build_saver(self):
        self.saver = tf.train.Saver(tf.trainable_variables())
       
    @timeit
    def initialization(self):
        self.sess.run(tf.initialize_all_variables())
      
    def lr_update(self):
        #exponential 
        if False:
            if self.epoch > 5 and self.global_step.eval() % 100 == 0:
                self.lr = self.lr * 0.96
                print("Updating learning rate to {}".format(self.lr))
        #adaptive
        if self.global_step.eval() % self.patience == 0 \
           and self.global_step.eval() / self.patience >= 1 \
           and np.min(self.losses[-self.patience:-1]) > self.best_loss:
            self.best_loss = np.min(self.losses[-self.patience:-1]) 
            self.lr = self.lr * 0.5
            print("Updating learning rate to {}".format(self.lr))
        
    @timeit
    def test_iter(self):
        test_data_size = len(open(self.test_source_data_path).readlines())
        N = int(math.ceil(test_data_size/self.batch_size))
        total_loss = 0
        for dsource, dtarget in self.iterator.train_batch():
            loss = self.sess.run([self.loss, self.summarizer],
                                 feed_dict={self.source: dsource,
                                            self.target: dtarget})
            total_loss += loss[0]
        total_loss /= N
        perplexity = np.exp(total_loss)
        return total_loss, perplexity
    
    @timeit
    def save(self): 
        self.saver.save(self.sess, 
                os.path.join(self.checkpoint_dir, self.name + "-" + self.dataset))
    
    #TODO: validation error every 100 iter 
    @timeit
    def train_iter(self, dsource, source_len, dtarget, target_len):
        output = self.sess.run([self.loss, self.optimizer, self.summarizer],
                feed_dict={self.learning_rate:  self.lr,
                           self.source:         dsource,
                           self.target:         dtarget,
                           self.source_len:     source_len,
                           self.target_len:     target_len
                           })
        self.losses.append(output[0])
        self.best_loss = min(self.best_loss, output[0])
        return output

    @timeit 
    def train_epoch(self, epoch):
        iterator = self.get_data_iterator()
        for dsource, source_len, dtarget, target_len in iterator:
            outputs = self.train_iter(dsource, source_len, dtarget, target_len) 
            step = self.global_step.eval() 
            self.writer.add_summary(outputs[-1], step)
            if step % 10 == 1:
                print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, step, outputs[0]))
            if step % 1000 == 1:
                self.save() 
                #test_loss, perplexity = self.test_iter()
                #print("Epoch: {}, Iteration: {}, Test loss: {}, Perplexity: {}".format(epoch, step, test_loss, perplexity))

            
    def build_model(self):
        print('-------------Variable building') 
        self.build_variables() 
        print('-------------Graph building') 
        self.build_graph()
        print('There are {} parameters in the graph.'.format(self.countParameters())) 
        print('-------------Variable initialization') 
        self.initialization()
        print('-------------Optimizer building') 
        self.build_optimizer() 
        print('-------------Saver, writer and summarizer building') 
        self.build_other_helpers() 
           
    def build_test_model(self):
        print('-------------Variable building') 
        self.build_variables() 
        print('-------------Graph building') 
        self.build_graph()
        print('There are {} parameters in the graph.'.format(self.countParameters())) 
        #print('-------------Optimizer building') 
        #self.build_optimizer() 
        print('-------------Saver') 
        self.build_saver() 
              
    def train(self):
        self.build_model()
        data_size = len(open(self.source_data_path).readlines())
        N = int(math.ceil(data_size/self.batch_size))
        for epoch in xrange(self.epochs):
            self.epoch = epoch 
            print("In epoch {}".format(epoch))
            self.train_epoch(epoch)
            # TODO: report: bleu, training cost, sample translation, alignment.
            # TODO: BLEU score sample decoder  
            self.lr_update() 

    def test(self):
        data_size = len(open(self.test_source_data_path).readlines())
        N = int(math.ceil(data_size/self.batch_size))
        total_loss = 0
        for dsource, dtarget in self.iterator.test_batch():
            if self.show: bar.next()
            loss = self.sess.run([self.loss],
                                 feed_dict={self.source: dsource,
                                            self.target: dtarget})
            total_loss += loss[0]
        if self.show:
            bar.finish()
            print("")
        total_loss /= N
        print(total_loss)
        perplexity = np.exp(total_loss)
        return perplexity
   
    @timeit
    def sample(self):
        pass

    @timeit
    def load(self, checkpointName = None):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        #else:
        #    raise Exception("[!] No checkpoint found")

    def countParameters(self): 
        total_parameters = 0
        for variable in tf.trainable_variables():
            print("Tensor name: {}; Shape: {}".format(variable.name, variable.get_shape())) 
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
                total_parameters += variable_parametes
        return total_parameters
