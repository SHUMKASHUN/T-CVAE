import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
from bert_serving.client import BertClient

def read_data(src_path):
    data_set = []
    counter = 0
    max_length1 = 0
    with tf.io.gfile.GFile(src_path, mode="r") as src_file:
        src = src_file.readline()
        while src:
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()

            sentences = []
            s = []
            for x in src.split(" "):
                id = int(x)
                if id != -1:
                    s.append(id)
                else:
                    if len(s) > max_length1:
                        max_length1 = len(s)
                    sentences.append(s)
                    s = []

            data_set.append(sentences)
            counter += 1
            src = src_file.readline()
    print(counter)
    print(max_length1)
    return data_set
    
def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def convert_whole_story(data,converter):
    word = []
    output = []
    pred_story = []
    for i in range (0,len(data)):
        for j in range (0,len(data[i])):
            for k in range (0,len(data[i][j])):
                word.append(tf.compat.as_str(converter[data[i][j][k]]))
            new_sentence = " ".join(word)
            #if j == i % 5:
            #    output.append(empty_string)
            #else:
            output.append(new_sentence)
            word.clear()
        new_stroy = " ".join(output)
        pred_story.append(new_stroy)
        output.clear()
    return pred_story

def convert_whole_sentence(data,converter):
    word = []
    output = []
    pred_story = []
    for i in range (0,len(data)):
        for j in range (0,len(data[i])):
            for k in range (0,len(data[i][j])):
                word.append(tf.compat.as_str(converter[data[i][j][k]]))
            new_sentence = " ".join(word)
            #if j == i % 5:
                
            #    output.append(empty_string)
            #else:
            pred_story.append(new_sentence)

            word.clear()
        new_stroy = " ".join(output)
        #pred_story.append(new_stroy)
        output.clear()
    return pred_story

empty_string = "                   "
def convert_partial_story(data,converter):
    word = []
    output = []
    pred_story = []
    for i in range (0,len(data)):
        for j in range (0,len(data[i])):
            for k in range (0,len(data[i][j])):
                word.append(tf.compat.as_str(converter[data[i][j][k]]))
            new_sentence = " ".join(word)
            if j == i % 5:
               output.append(empty_string)
            else:
            output.append(new_sentence)
            word.clear()
        new_stroy = " ".join(output)
        pred_story.append(new_stroy)
        output.clear()
    return pred_story


def main():
    train_data = read_data("../data/train.ids")
    valid_data = read_data("../data/valid.ids")
    to_vocab, rev_to_vocab = initialize_vocabulary("../data/vocab_20000")
    train_data = train_data[0:78016]
    valid_data = train_data[0:9792]
    #FOR Train dataset
    train_story = convert_whole_story(train_data,rev_to_vocab)
    valid_story = convert_whole_story(valid_data,rev_to_vocab)
    #train_sentence = convert_whole_sentence(train_data,rev_to_vocab)
    #valid_sentence = convert_whole_sentence(valid_data,rev_to_vocab)
    train_story_prior = convert_partial_story(train_data,rev_to_vocab)
    valid_story_prior = convert_partial_story(valid_data,rev_to_vocab)

    bc = BertClient()
    ####################### Train post ###################
    train_post = bc.encode(train_story[0:256])
    for i in range (1,304):
        tmp = bc.encode(train_story[i*256:256+i*256])
        train_post = np.concatenate((train_post,tmp))
        if (i % 10 == 0):
            print("train post %d done"%(i))
    tmp = bc.encode(train_story[77824:78016])
    train_post = np.concatenate((train_post,tmp))
    print(train_post.shape) #should be [78016,105,512]
    np.savez("train_post.npz",train_post)
    print("train post encode done")
    ####################### Train prior ###################
    train_prior = bc.encode(train_story_prior[0:256])
    for i in range (1,304):
        tmp = bc.encode(train_story_prior[i*256:256+i*256])
        train_prior = np.concatenate((train_prior,tmp))
        if (i % 10 == 0):
            print("train prior %d done"%(i))
    tmp = bc.encode(train_story_prior[77824:78016])
    train_prior = np.concatenate((train_prior,tmp))
    print(train_prior.shape) #should be [78016,105,512]
    np.savez("train_prior.npz",train_prior)
    print("train prior encode done")
#####################################################    
    valid_post = bc.encode(valid_story[0:256])
    for i in range (1,38):
        tmp = bc.encode(valid_story[i*256:256+i*256])
        valid_post = np.concatenate((valid_post,tmp))
        if (i % 5 == 0):
            print("valid post %d done"%(i))
    tmp = bc.encode(valid_story[9728:9792])
    valid_post = np.concatenate((valid_post,tmp))
    print(valid_post.shape) #should be [9792,105,512]
    np.savez("valid_post.npz",valid_post)
    print("valid post encode done")
#######################################################
    valid_prior = bc.encode(valid_story_prior[0:256])
    for i in range (1,38):
        tmp = bc.encode(valid_story_prior[i*256:256+i*256])
        valid_prior = np.concatenate((valid_prior,tmp))
        if (i % 5 == 0):
            print("valid prior %d done"%(i))
    tmp = bc.encode(valid_story_prior[9728:9792])
    valid_prior = np.concatenate((valid_prior,tmp))
    print(valid_prior.shape) #should be [9792,105,512]
    np.savez("valid_prior.npz",valid_prior)
    print("valid prior encode done")
if __name__ == "__main__":
    main()