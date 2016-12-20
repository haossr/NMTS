from __future__ import division
from __future__ import print_function
from random import shuffle

def pre_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[(max_len - len(lst)):] = lst
    return nlst

def post_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[:len(lst)] = lst
    return nlst

def read_vocabulary(data_path):
    #Check '<unk>', '<s>', '</s>', '<pad>'
    vocab = {w:i for i,w in enumerate(open(data_path).read().splitlines())}
    for token in ['<unk>', '<s>', '</s>', '<pad>']:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def data_iterator(source_data_path,
                  target_data_path,
                  source_vocab,
                  target_vocab,
                  max_size,
                  batch_size):
    with open(source_data_path, "rb") as f_in, open(target_data_path) as f_out:
        prev_batch = 0
        data_in    = []
        data_out   = []
        for i, (lin, lout) in enumerate(zip(f_in, f_out)):
            if i - prev_batch >= batch_size:
                prev_batch = i
                yield data_in, data_out
                data_in  = []
                data_out = []
            in_text = [source_vocab[w] if w in source_vocab else source_vocab["<unk>"]
                       for w in lin.replace("\n", "").split(" ")][:max_size][::-1]
            out_text = [target_vocab[w] if w in target_vocab else target_vocab["<unk>"]
                        for w in lout.replace("\n", " " + "</s>")
                        .split(" ")][:max_size-1]
            out_text = [target_vocab["<s>"]] + out_text
            data_in.append(pre_pad(in_text, source_vocab["<pad>"], max_size))
            data_out.append(post_pad(out_text, target_vocab["<pad>"], max_size))
        if (i + 1) % batch_size == 0:
            yield data_in, data_out

def data_iterator_len(source_data_path,
                      target_data_path,
                      source_vocab,
                      target_vocab,
                      max_size,
                      batch_size):
    with open(source_data_path, "rb") as f_in, open(target_data_path) as f_out:
        prev_batch = 0
        data_in    = []
        data_out   = []
        len_in     = []
        len_out    = []
        for i, (lin, lout) in enumerate(zip(f_in, f_out)):
            if i - prev_batch >= batch_size:
                prev_batch = i
                yield data_in, len_in, data_out, len_out
                data_in  = []
                data_out = []
                len_in   = []
                len_out  = []
            in_text = [source_vocab[w] if w in source_vocab else source_vocab["<unk>"]
                       for w in lin.replace("\n", "").split(" ")][:max_size][::-1]
            out_text = [target_vocab[w] if w in target_vocab else target_vocab["<unk>"]
                        for w in lout.replace("\n", " " + "</s>")
                        .split(" ")][:max_size-1]
            out_text = [target_vocab["<s>"]] + out_text
            len_in.append(len(in_text))
            len_out.append(len(out_text))
            data_in.append(post_pad(in_text, source_vocab["<pad>"], max_size))
            data_out.append(post_pad(out_text, target_vocab["<pad>"], max_size))

        if (i + 1) % batch_size == 0:
            yield data_in, len_in, data_out, len_out

def sort_data_files(source_data_path, target_data_path):
    words = [len(line.split(" ")) for line in open(source_data_path, "rb").readlines()]
    indices = sorted(xrange(len(words)), key=lambda k: words[k])
    source_lines = open(source_data_path, "rb").readlines()
    target_lines = open(target_data_path, "rb").readlines()
    with open(source_data_path + ".sorted", "wb") as f:
        f.write("".join([source_lines[i] for i in indices]))
    with open(target_data_path + ".sorted", "wb") as f:
        f.write("".join([target_lines[i] for i in indices]))

def batch_shuffle(source_data_path, target_data_path, batch_size):
    source = open(source_data_path, "rb").readlines()
    target = open(target_data_path, "rb").readlines()
    source_batches = [source[i:i+batch_size] for i in xrange(0, len(source), batch_size)]
    target_batches = [target[i:i+batch_size] for i in xrange(0, len(target), batch_size)]
    indices = [i for i in xrange(len(source_batches))]
    shuffle(indices)
    with open(source_data_path + ".shuffled", "wb") as f:
        f.write("".join(["".join(line) for i in indices for line in source_batches[i]]))
    with open(target_data_path + ".shuffled", "wb") as f:
        f.write("".join(["".join(line) for i in indices for line in target_batches[i]]))

def prune_sentence_length(source_data_path, target_data_path, max_size):
    source = open(source_data_path, "rb").readlines()
    target = open(target_data_path, "rb").readlines()
    with open(source_data_path + ".pruned", "wb") as f_s, open(target_data_path + ".pruned", "wb") as f_t:
        for ls, lt in zip(source, target):
            if len(ls.split(" ")) < max_size and len(lt.split(" ")) < max_size:
                f_s.write(ls)
                f_t.write(lt)

class DataIter():
    def __init__(self, 
                 train_source_data_path,
                 train_target_data_path,
                 source_vocab_path,
                 target_vocab_path,
                 max_size               = 30,
                 batch_size             = 128,
                 valid_source_data_path = None,
                 valid_target_data_path = None,
                 test_source_data_path  = None,
                 test_target_data_path  = None):
        self.train_source_data_path = train_source_data_path
        self.train_target_data_path = train_target_data_path
        self.source_vocab_path      = source_vocab_path
        self.target_vocab_path      = target_vocab_path
        self.valid_source_data_path = valid_source_data_path
        self.valid_target_data_path = valid_target_data_path
        self.test_source_data_path  = test_source_data_path
        self.test_target_data_path  = test_target_data_path
        self.source_vocab           = read_vocabulary(self.source_vocab_path) 
        self.target_vocab           = read_vocabulary(self.target_vocab_path) 
        self.max_size               = max_size
        self.batch_size             = batch_size
        
        self.source_vocab_size  = len(self.source_vocab)
        self.target_vocab_size  = len(self.target_vocab)
        self.train_size         = len(open(self.train_source_data_path).readlines())
        self.valid_size         = len(open(self.valid_source_data_path).readlines())
        self.test_size          = len(open(self.test_source_data_path).readlines())
    
    def test(self):
        print("Testing iterator...")
        print("Train_source_data_path: {}".format(self.train_source_data_path)) 
        print("Train_target_data_path: {}".format(self.train_target_data_path)) 
        print("Valid_source_data_path: {}".format(self.valid_source_data_path)) 
        print("Valid_target_data_path: {}".format(self.valid_target_data_path)) 
        print("Test_source_data_path: {}".format(self.test_source_data_path)) 
        print("Test_target_data_path: {}".format(self.test_target_data_path)) 
        print("max_size: {}".format(self.max_size)) 
        print("batch_size: {}".format(self.batch_size)) 
        print("Source vocabulary: {}".format(len(self.source_vocab))) 
        print("Target vocabulary: {}".format(len(self.target_vocab)))
        print("Number of iterations in an epoch: {}".format(len(list(self.train_batch()))))
   
    def batch_shuffle(self, batch_size = None):
        if batch_size == None: batch_size = self.batch_size
        batch_shuffle(self.train_source_data_path, train_target_data_path, batch_size)

    def train_batch(self):
        return data_iterator_len(self.train_source_data_path,
                      self.train_target_data_path,
                      self.source_vocab,
                      self.target_vocab,
                      self.max_size,
                      self.batch_size) 
    
    def valid_batch(self):
        return data_iterator_len(self.valid_source_data_path,
                      self.valid_target_data_path,
                      self.source_vocab,
                      self.target_vocab,
                      self.max_size,
                      self.batch_size) 
    
    def test_batch(self):
        return data_iterator_len(self.test_source_data_path,
                      self.test_target_data_path,
                      self.source_vocab,
                      self.target_vocab,
                      self.max_size,
                      self.batch_size) 
    
