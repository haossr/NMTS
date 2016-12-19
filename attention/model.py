from __future__ import division
from __future__ import print_function

from datetime import datetime
from data import data_iterator, data_iterator_len
from data import read_vocabulary
from utils import ProgressBar
from bleu.length_analysis import process_files

import tensorflow as tf
import numpy as np
import math
import os
import time
import collections

class AttentionNN(object):
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
        return "attention-{}-{}-{}-{}-{}".format(self.dataset, date.month, date.day, date.hour, date.minute)
    
    def get_data_iterator(self):
        return data_iterator_len(self.source_data_path,
                                 self.target_data_path,
                                 self.source_vocab,
                                 self.target_vocab,
                                 self.max_size, 
                                 self.batch_size)
    
    def get_testdata_iterator(self):
        return data_iterator_len(self.test_source_data_path,
                                 self.test_target_data_path,
                                 self.source_vocab,
                                 self.target_vocab,
                                 self.max_size, 
                                 self.batch_size)
     
    def __init__(self, config, sess):
        print('Reading config file...') 
        self.sess          = sess
       
        #Model: main parameters
        self.emb_size      = config.emb_size
        self.hidden_size   = config.hidden_size
        self.num_layers    = config.num_layers
        self.max_size      = config.max_size
        self.epochs        = config.epochs
        self.s_nwords      = config.s_nwords
        self.t_nwords      = config.t_nwords
        self.minval        = config.minval
        self.maxval        = config.maxval
        
        #Model: regularization parameters
        self.dropout       = config.dropout 
        
        #Tensorflow: main parameters
        self.lr             = config.lr_init
        self.batch_size     = config.batch_size
        self.max_grad_norm  = config.max_grad_norm
        self.optimizer_name = config.optimizer_name
        self.patience       = config.patience 
        self.global_step    = None 
        self.epoch          = None 
        self.loss           = None
        self.losses         = []
        self.best_loss      = float('inf') 
        
        #Tensorflow: modules
        self.writer     = None
        self.summarizer = None
        self.optimizer  = None
        self.saver      = None

        #Data
        self.dataset                = config.dataset
        self.source_data_path       = config.source_data_path
        self.target_data_path       = config.target_data_path
        self.test_source_data_path  = config.test_source_data_path
        self.test_target_data_path  = config.test_target_data_path
        self.source_vocab_path      = config.source_vocab_path
        self.target_vocab_path      = config.target_vocab_path
        self.prediction_data_path   = config.prediction_data_path 
        self.checkpoint_dir         = config.checkpoint_dir
        self.source_vocab           = read_vocabulary(self.source_vocab_path) 
        self.target_vocab           = read_vocabulary(self.target_vocab_path) 
        
        self.show = config.show
        
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Directory {} not found".format(self.checkpoint_dir))

    ############################################################################
    #Tensorflow model 
    ############################################################################
    @timeit
    def build_variables(self): 
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = tf.placeholder(tf.float32, shape=[])#lr = tf.Variable(self.lr_init, trainable=False, name="learning_rate")
        initializer = tf.random_uniform_initializer(self.minval, self.maxval) 
        
        print('1.Input layer') 
        with tf.variable_scope("input"):
            self.source     = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="source")
            self.target     = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="target")
            self.source_len = tf.placeholder(tf.int32, [self.batch_size], name="source_length")
            self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_length")
        
        print('2.Embedding layer') 
        with tf.variable_scope("embed"):
            self.s_emb      = tf.get_variable("s_embedding",
                                shape       = [self.s_nwords, self.emb_size],
                                initializer = initializer) 
            self.s_proj_W   = tf.get_variable("s_proj_W", 
                                shape       = [self.emb_size, self.hidden_size],
                                initializer = initializer)
            self.s_proj_b   = tf.get_variable("s_proj_b", 
                                shape=[self.hidden_size],
                                initializer=initializer)
            self.t_emb      = tf.get_variable("t_embedding",
                                shape       = [self.t_nwords, self.emb_size],
                                initializer = initializer)
            self.t_proj_W   = tf.get_variable("t_proj_W", 
                                shape=[self.emb_size, self.hidden_size],
                                initializer=initializer)
            self.t_proj_b   = tf.get_variable("t_proj_b", 
                                shape=[self.hidden_size],
                                initializer=initializer)
        
        print('3.Encoding layer') 
        with tf.variable_scope("encoder"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
        
        print('4.Decoding layer') 
        with tf.variable_scope("decoder"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        print('5.Output layer')
        with tf.variable_scope("proj"):
            self.proj_W  = tf.get_variable("W", 
                                shape       = [self.hidden_size, self.emb_size],
                                initializer = initializer)
            self.proj_b  = tf.get_variable("b", 
                                shape       = [self.emb_size],
                                initializer = initializer)
            self.proj_Wo = tf.get_variable("Wo", 
                                shape       = [self.emb_size, self.t_nwords],
                                initializer = initializer)
            self.proj_bo = tf.get_variable("bo", 
                                shape       = [self.t_nwords],
                                initializer = initializer)

        print('6.Attention layer')
        with tf.variable_scope("attention"):
            self.v_a = tf.Variable(tf.random_uniform([self.hidden_size, 1],
                                   minval=self.minval, maxval=self.maxval), name="v_a")
            self.W_a = tf.Variable(tf.random_uniform([2*self.hidden_size, self.hidden_size],
                                   minval=self.minval, maxval=self.maxval), name="W_a")
            self.b_a = tf.Variable(tf.random_uniform([self.hidden_size],
                                   minval=self.minval, maxval=self.maxval), name="b_a")
            self.W_c = tf.Variable(tf.random_uniform([2*self.hidden_size, self.hidden_size],
                                       minval=self.minval, maxval=self.maxval), name="W_c")
            self.b_c = tf.Variable(tf.random_uniform([self.hidden_size],
                                       minval=self.minval, maxval=self.maxval), name="b_a")
            
            
            self.W_g = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size],
                                       minval=self.minval, maxval=self.maxval), name="W_g")
             
    @timeit
    def build_graph(self):
        print('1.Embedding layer') 
        with tf.variable_scope("embed"):
            source_xs = tf.nn.embedding_lookup(self.s_emb, self.source)
            source_xs = tf.split(1, self.max_size, source_xs)
            target_xs = tf.nn.embedding_lookup(self.t_emb, self.target)
            target_xs = tf.split(1, self.max_size, target_xs)
            
            
        print('2.Encoding layer') 
        initial_state = self.encoder.zero_state(self.batch_size, tf.float32)
        s = initial_state
        encoder_hs = []
        with tf.variable_scope("encoder"):
            for t in xrange(self.max_size):
                x = tf.squeeze(source_xs[t], [1])
                x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
                if t > 0: tf.get_variable_scope().reuse_variables()
                hs = self.encoder(x, s)
                s = hs[1]
                h = hs[0]
                encoder_hs.append(h)

        logits     = []
        probs      = []
        print('3.Decoding and loss layer') 
        # s is now final encoding hidden state
        with tf.variable_scope("decoder"):
            for t in xrange(self.max_size):
                x = tf.squeeze(target_xs[t], [1])
                x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
                if t > 0: tf.get_variable_scope().reuse_variables()
                hs = self.decoder(x, s)
                s = hs[1]
                h = hs[0]

                #Concating score
                #scores = [tf.matmul(tf.tanh(tf.batch_matmul(tf.concat(1, [h, h_s]),\
                #                                            self.W_a) + self.b_a),\
                #                                            self.v_a)
                #          for h_s in encoder_hs]
                #Dot score 
                scores = [tf.reduce_sum(tf.mul(h, h_s),1) for h_s in encoder_hs]  
                scores = tf.nn.softmax(tf.reshape(tf.pack(scores),
                                       [self.batch_size, self.max_size]))
                #General score

                #Attention layer 
                c_t = tf.reduce_sum([a_s*h_s for a_s, h_s in
                                     zip(tf.split(1, self.max_size, scores),
                                         encoder_hs)], 0)
                h_t = tf.batch_matmul(tf.concat(1, [h, c_t]), self.W_c) + self.b_c
                
                outemb = tf.batch_matmul(h_t, self.proj_W) + self.proj_b
                logit = tf.batch_matmul(outemb, self.proj_Wo) + self.proj_bo
                prob  = tf.nn.softmax(logit)
                logits.append(logit)
                probs.append(prob)

        logits     = logits[:-1]
        targets    = tf.split(1, self.max_size, self.target)[1:]
        weights    = tf.unpack(tf.sequence_mask(lengths = self.target_len, 
                                      maxlen  = self.max_size-1,
                                      dtype   = tf.float32), None, 1)
         
        #weights   = [tf.ones([self.batch_size]) for _ in xrange(self.max_size - 1)]
        self.loss  = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
        self.probs = tf.transpose(tf.pack(probs), [1, 0, 2])
    
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
        tf.scalar_summary("learning rate", self.learning_rate)
        #tf.scalar_summary("best training loss", self.best_loss)
        self.summarizer = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("./logs/{}".format(self.get_log_name()),\
                 self.sess.graph)

       
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
        iterator = data_iterator(self.test_source_data_path,
                                 self.test_target_data_path,
                                 self.source_vocab,
                                 self.target_vocab,
                                 self.max_size, 
                                 self.batch_size)
        total_loss = 0
        for dsource, dtarget in iterator:
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
                os.path.join(self.checkpoint_dir, self.get_log_name()))
    
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
        print('-------------Optimizer building') 
        self.build_optimizer() 
        print('-------------Saver, writer and summarizer building') 
        self.build_other_helpers() 
        print('-------------Variable initialization') 
        self.initialization()
   
    def build_test_model(self):
        print('-------------Variable building') 
        self.build_variables() 
        print('-------------Graph building') 
        self.build_graph()
        print('There are {} parameters in the graph.'.format(self.countParameters())) 
        #print('-------------Optimizer building') 
        #self.build_optimizer() 
        print('-------------Saver, writer and summarizer building') 
        self.build_other_helpers() 
              
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

        iterator = data_iterator(self.test_source_data_path,
                                 self.test_target_data_path,
                                 self.source_vocab,
                                 self.target_vocab,
                                 self.max_size, 
                                 self.batch_size)
        total_loss = 0
        for dsource, dtarget in iterator:
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
        self.build_test_model() 
        self.load()
        iterator = self.get_testdata_iterator()
        inv_source_vocab = {v:k for k,v in self.source_vocab.iteritems()} 
        inv_target_vocab = {v:k for k,v in self.target_vocab.iteritems()} 
        samples  = []
        for dsource, source_len, _, _ in iterator:
            print("###########################################################") 
            print("Source sentence") 
            print("###########################################################") 
            for ds in dsource:
                print(" ".join([inv_source_vocab[i] for i in reversed(ds) if inv_source_vocab[i] != "<pad>"]))
            psuedo_target_len = [self.max_size - 1 for _ in xrange(self.batch_size)]
            #psuedo_target_len = [[self.target_vocab["<s>"]] + [self.target_vocab["<pad>"]] * (self.max_size-1)]
            dtarget = [[self.target_vocab["<pad>"]] * self.max_size]
            psuedo_dtarget = dtarget * self.batch_size
            
            probs,  = self.sess.run([self.probs], 
                        feed_dict = {self.source: dsource,
                        self.target:         psuedo_dtarget,
                        self.source_len:     source_len,
                        self.target_len:     psuedo_target_len})
            print("###########################################################") 
            print("Target sentence") 
            print("###########################################################") 
            with open(self.prediction_data_path, 'w+') as prediction_file:
                for j in range(self.batch_size):
                    target_probs    = probs[j]
                    target_indices  = np.argmax(target_probs, 1)
                    target_sentence = []
                    k = 0
                    for i in target_indices:
                        next_word = inv_target_vocab[i]
                        if next_word != "</s>":
                            target_sentence.append(next_word)
                        else:
                            break
                    print(" ".join(target_sentence))
                    print(" ".join(target_sentence), file=prediction_file)
            process_files(self.prediction_data_path, self.target_data_path) 
            
    @timeit
    def load(self):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("[!] No checkpoint found")

    def countParameters(self): 
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
                total_parameters += variable_parametes
        return total_parameters
