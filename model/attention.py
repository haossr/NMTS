from __future__ import division
from __future__ import print_function

import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from datetime import datetime
from bleu.length_analysis import process_files
from model import Model
from utils import *
import tensorflow as tf
import numpy as np
import math
import os
import time
import collections

class AttentionNN(Model):
    def __init__(self, config, sess):
        print('Reading config file...') 
        self.sess          = sess
       
        #Model: main parameters
        self.name          = "attention"
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
        self.valid_losses   = []
        self.best_loss      = float('inf') 
        
        #Tensorflow: modules
        self.writer     = None
        self.summarizer = None
        self.optimizer  = None
        self.saver      = None

        #Data
        self.iterator       = config.iterator 
        self.checkpoint_dir = config.checkpoint_dir + "/" + self.name + "/"
        self.checkpointName = config.checkpointName 
        self.dataset        = config.dataset
        self.train_size     = self.iterator.train_size  
        self.valid_size     = self.iterator.valid_size  
        self.test_size      = self.iterator.test_size  
        
        self.prediction_data_path = "predict"
        self.truth_data_path = "truth"
        self.is_test         = config.is_test
        
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Directory {} not found".format(self.checkpoint_dir))

    @timeit
    def build_variables(self): 
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.valid_loss = tf.placeholder(tf.float32, shape=[])
        self.bleu = tf.placeholder(tf.float32, shape=[])
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
                h, s = self.encoder(x, s)
                encoder_hs.append(h)
        
        print('3.Decoding and loss layer') 
        with tf.variable_scope("decoder"):
            logits     = []
            probs      = []
            s = self.decoder.zero_state(self.batch_size, tf.float32) 
            for t in xrange(self.max_size):
                x = tf.squeeze(target_xs[t], [1])
                x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
                if t > 0: tf.get_variable_scope().reuse_variables()
                h, s = self.decoder(x, s)

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
            weights    = tf.unpack(tf.to_float(tf.less(tf.expand_dims(tf.range(0, self.max_size-1, 1), 1), 
                                    tf.expand_dims(self.target_len, 0))
                            ), None, 0)
            
            #TF11 
            #weights    = tf.unpack(tf.sequence_mask(lengths = self.target_len, 
            #                              maxlen  = self.max_size-1,
            #                              dtype   = tf.float32), None, 1)
            self.loss  = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
            self.probs = tf.transpose(tf.pack(probs), [1, 0, 2])
    
        print('4.Decoding and loss layer (testing)') 
        with tf.variable_scope("decoder", reuse=True):
            logits     = []
            probs      = []
            s = self.decoder.zero_state(self.batch_size, tf.float32) 
            for t in xrange(self.max_size):
                if t == 0: 
                    x = tf.squeeze(target_xs[t], [1])
                x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
                if t > 0: tf.get_variable_scope().reuse_variables()
                h, s = self.decoder(x, s)

                scores = [tf.reduce_sum(tf.mul(h, h_s),1) for h_s in encoder_hs]
                scores = tf.nn.softmax(tf.reshape(tf.pack(scores),
                                       [self.batch_size, self.max_size]))
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

                x = tf.cast(tf.argmax(prob, 1), tf.int32)
                x = tf.nn.embedding_lookup(self.t_emb, x)

            logits     = logits[:-1]
            targets    = tf.split(1, self.max_size, self.target)[1:]
            weights    = tf.unpack(tf.to_float(tf.less(tf.expand_dims(tf.range(0, self.max_size-1, 1), 1), 
                                    tf.expand_dims(self.target_len, 0))
                            ), None, 0)
            
            #TF11 
            #weights    = tf.unpack(tf.sequence_mask(lengths = self.target_len, 
            #                              maxlen  = self.max_size-1,
            #                              dtype   = tf.float32), None, 1)
            self.loss_test  = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
            self.probs_test = tf.transpose(tf.pack(probs), [1, 0, 2])
    
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
        #self.summarizer = tf.merge_all_summaries() 
        self.summarizer = tf.merge_summary([tf.scalar_summary("Learning Rate", self.learning_rate),
                                            tf.scalar_summary("Training Loss", self.loss)])
        
        self.test_summarizer = tf.merge_summary([tf.scalar_summary("Validation Loss", self.valid_loss), 
                                                 tf.scalar_summary("BLEU", self.bleu)])
        self.writer = tf.train.SummaryWriter("./logs/{}".format(self.get_log_name()),
                     self.sess.graph)

    def lr_update(self):
        #exponential 
        if True:
            if self.epoch > 5 and self.global_step.eval() % 5000 == 0:
                self.lr = self.lr * 0.5
                print("Updating learning rate to {}".format(self.lr))
        #adaptive
        if False:
            if self.global_step.eval() % self.patience == 0 \
               and self.global_step.eval() / self.patience >= 1 \
               and np.min(self.losses[-self.patience:-1]) > self.best_loss:
                self.best_loss = np.min(self.losses[-self.patience:-1]) 
                self.lr = self.lr * 0.5
                print("Updating learning rate to {}".format(self.lr))
        
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
        for dsource, source_len, dtarget, target_len in self.iterator.train_batch():
            outputs = self.train_iter(dsource, source_len, dtarget, target_len) 
            step = self.global_step.eval() 
            self.writer.add_summary(outputs[-1], step)
            if step % 10 == 1:
                print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, step, outputs[0]))
            if step % 100 == 1:
                valid_loss = self.test()
                bleu = self.sample()
                step = self.global_step.eval() 
                self.writer.add_summary(self.sess.run([self.test_summarizer], 
                    {self.valid_loss: valid_loss, self.bleu: bleu})[0], step)
                print("Epoch: {}, Iteration: {}, Validation Loss: {}, BLEU: {}".format(epoch, step, valid_loss, bleu))
            if step % 1000 == 1 and step > 1000: 
                self.save() 
              
    def train(self):
        if not self.saver:
            self.build_model()
        N = int(math.ceil(self.train_size/self.batch_size))
        for epoch in xrange(self.epochs):
            self.epoch = epoch 
            print("In epoch {}".format(epoch))
            self.train_epoch(epoch)
            self.lr_update()
            #Early stop
            step = self.global_step.eval() 
            if step > 20 * self.patience and np.min(self.valid_losses[-20*self.patience:-1]) > np.min(self.valid_losses):
                print("EARLY STOP") 
                break
   
    @timeit 
    def test(self):
        N = int(math.ceil(self.test_size/self.batch_size))
        valid_loss = 0
        for dsource, source_len, dtarget, target_len in self.iterator.test_batch():
            loss = self.sess.run([self.loss],
                                 feed_dict = {self.source:         dsource,
                                              self.target:         dtarget,
                                              self.source_len:     source_len,
                                              self.target_len:     target_len
                                   })
            valid_loss += loss[0]
        valid_loss /= N
        self.valid_losses.append(valid_loss)
        return valid_loss 
   
    @timeit
    def sample(self, verbose=False):
        if self.is_test: verbose = True
        if os.path.exists(self.prediction_data_path): os.remove(self.prediction_data_path) 
        if os.path.exists(self.truth_data_path): os.remove(self.truth_data_path) 
        inv_source_vocab = {v:k for k,v in self.iterator.source_vocab.iteritems()} 
        inv_target_vocab = {v:k for k,v in self.iterator.target_vocab.iteritems()} 
        #print(inv_target_vocab)
        samples  = []
        for dsource, source_len, dtarget, target_len in self.iterator.valid_batch():
            if verbose:
                print("###########################################################") 
                print("Source sentence") 
                print("###########################################################") 
            with open(self.truth_data_path, 'a') as truth_file:
                for ds, dt in zip(dsource, dtarget):
                    if verbose: print(" ".join([inv_source_vocab[i] for i in reversed(ds) if inv_source_vocab[i] != "<pad>"]))
                    print(" ".join(reversed([inv_target_vocab[i] for i in reversed(dt) if inv_target_vocab[i] not in ["<pad>", "<s>", "</s>"]])), file=truth_file)
            psuedo_target_len = [self.max_size - 1 for _ in xrange(self.batch_size)]
            #psuedo_target_len = [[self.target_vocab["<s>"]] + [self.target_vocab["<pad>"]] * (self.max_size-1)]
            
            dtarget = [[self.iterator.target_vocab["<s>"]] + [self.iterator.target_vocab["<pad>"]] * (self.max_size-1)]
            psuedo_dtarget = dtarget * self.batch_size
            
            probs,  = self.sess.run([self.probs_test], 
                        feed_dict = {self.source: dsource,
                        self.target:         psuedo_dtarget,
                        self.source_len:     source_len,
                        self.target_len:     psuedo_target_len})
            if verbose:
                print("###########################################################") 
                print("Target sentence") 
                print("###########################################################") 
            with open(self.prediction_data_path, 'a') as prediction_file:
                for j in xrange(self.batch_size):
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
                    print(" ".join(target_sentence), file=prediction_file)
        bleu = 0
        try:
            bleu = process_files(self.prediction_data_path, self.truth_data_path, True) 
        except Exception:
            pass
        return bleu 
    
    @timeit
    def load(self):
        self.build_model() 
        if self.checkpointName:
            print("[*] Reading checkpoints...")
            new_saver = tf.train.import_meta_graph(self.checkpoint_dir + self.checkpointName + '.meta')
            new_saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))
        else:   
            print("[*] Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("[!] No checkpoint found")
