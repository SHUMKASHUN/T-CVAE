from tensorflow.python.platform import gfile
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

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

def get_pred(path):
    prediction=[]
    file = open(path,'r') #'./output/predict2_file105000'
    line = file.read().splitlines()
    while line:
        prediction.append(line)
        line = file.read().splitlines()
    file.close()
    return prediction  #Using prediction[0][i] to see each sentence

empty_string = "               "
def convert_pred_to_story(data,converter):
    word = []
    output = []
    pred_story = []
    for i in range (0,78016):
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

def get_bert_post_output(indicate_id):
    train_data = read_data("data/train.ids")
    to_vocab, rev_to_vocab = initialize_vocabulary("data/vocab_20000")
    train_story = convert_pred_to_story(train_data,rev_to_vocab)
    bc = BertClient()
    #max indicate_id = 1218
    result = bc.encode(train_story[(0+indicate_id * 64) :(64+indicate_id * 64)])
    return result

