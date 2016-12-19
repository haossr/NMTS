import data

def test_iterator(source_data_path, 
                  target_data_path, 
                  source_vocab,
                  target_vocab,
                  max_size, 
                  batch_size ):
    print("Testing iterator...")
    print("source_data_path: {}".format(source_data_path)) 
    print("target_data_path: {}".format(target_data_path)) 
    print("max_size: {}".format(max_size)) 
    print("batch_size: {}".format(batch_size)) 
    print("Source vocabulary: {}".format(len(source_vocab))) 
    print("Target vocabulary: {}".format(len(target_vocab)))
    it = data.data_iterator_len(source_data_path, target_data_path, source_vocab, target_vocab, max_size, batch_size)
    print("Number of iterations in an epoch: {}".format(len(list(it))))
if __name__ == "__main__":
    test_iterator(source_data_path, target_data_path, source_vocab, target_vocab, max_size, batch_size)
