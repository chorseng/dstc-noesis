import os

import json
import ijson
import functools

import tensorflow as tf

tf.flags.DEFINE_integer(
    "min_word_frequency", 1, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string("train_in", None, "Path to input data file")
tf.flags.DEFINE_string("validation_in", None, "Path to validation data file")

tf.flags.DEFINE_string("train_out", None, "Path to output train tfrecords file")
tf.flags.DEFINE_string("validation_out", None, "Path to output validation tfrecords file")

tf.flags.DEFINE_string("vocab_path", None, "Path to save vocabulary txt file")
tf.flags.DEFINE_string("vocab_processor", None, "Path to save vocabulary processor")

FLAGS = tf.flags.FLAGS

#TRAIN_PATH = os.path.join(FLAGS.train_in)
#VALIDATION_PATH = os.path.join(FLAGS.validation_in)

TRAIN_PATH = '/home/chorseng/fashion_data/dialogs/train/fashion_train_dials.json'
VALIDATION_PATH = '/home/chorseng/fashion_data/dialogs/valid/fashion_dev_dials.json'
TRAIN_CANDIDATE_PATH = '/home/chorseng/fashion_data/dialogs/train/fashion_train_dials_retrieval_candidates.json'
VALIDATION_CANDIDATE_PATH = '/home/chorseng/fashion_data/dialogs/train/fashion_dev_dials_retrieval_candidates.json'

def combine_data(dialogs, candidates, mode):
    raw_data = []
    idx1 = -1

    for dialog in dialogs:
        idx2 = -1
        idx1+=1
        for dialog_turn in dialog:
            idx2+=1
            dial_dict = {}
            dial_dict['data_split'] = mode
            dial_dict['domain'] = 'fashion'
            dial_dict['example-id'] = str(dialog['dialogue_idx']) + '-' + str(dialog_turn['turn_idx'])
            dial_dict['messages-so-far'] = [{'speaker' : USER, 'utterance': dialog['transcript']}]
            
            correct_candidate_idx = candidates['retrieval_candidates'][idx1]['retrieval_candidates'][idx2]['retrieval_candidates'][0]
            correct_candidate_response = candidates['system_transcript_pool'][int(correct_candidate_idx)]      
            dial_dict['options-for-correct-answers'] = [{'candidate-id': correct_candidate_idx, 'utterance': correct_candidate_response}]
            
            dial_dict['options-for-next'] = []
            for candidate_response in candidates['retrieval_candidates'][idx1]['retrieval_candidates'][idx2]['retrieval_candidates']:
                candidate_dict = {'candidate-id': candidate_response, 'utterance': candidates['system_transcript_pool'][int(candidate_response)]}
                dial_dict['options-for-next'].append(candidate_dict)
            
            raw_data.append(dial_dict)
    
    return raw_data



def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


def process_dialog(dialog):
    """
    Add EOU and EOT tags between utterances and create a single context string.
    :param dialog:
    :return:
    """

    row = []
    utterances = dialog['messages-so-far']                            
    #utterances = dialog['transcript']

    # Create the context
    context = ""
    #speaker = USER
    #context += utterance + "__eou__"
    
    
    speaker = None
    for msg in utterances:
        if speaker is None:
            context += msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        elif speaker != msg['speaker']:
            context += "__eot__ " + msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        else:
            context += msg['utterance'] + " __eou__ "

    context += "__eot__"
    row.append(context)

    # Create the next utterance options and the target label
    correct_answer = dialog['options-for-correct-answers'][0]
    #correct_answer = dialog['system_transcript']
    #for idx, response in enumerate(response_pool):
    #    if response == correct_answer:
    #        target_id = idx
    target_id = correct_answer['candidate-id']
    
    target_index = None
    for i, utterance in enumerate(dialog['options-for-next']):
        if utterance['candidate-id'] == target_id:
            target_index = i
        row.append(utterance['utterance'] + " __eou__ ")

    if target_index is None:
        print('Correct answer not found in options-for-next - example {}. Setting 0 as the correct index'.format(dialog['example-id']))
    else:
        row.append(target_index)

    return row


def create_dialog_iter(filename):
    """
    Returns an iterator over a JSON file.
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        json_data = ijson.items(f, 'item')
        # iterating through conversations
        #for entry in json_data['dialogue_data']:
        for entry in json_data:
            row = process_dialog(entry)
            yield row
            # iterating through rounds of conversations
            #for dial in entry['dialogue']:
            #    row = process_dialog(dial)
            #    yield row

def create_utterance_iter(input_iter):
    """
    Returns an iterator over every utterance (context and candidates) for the VocabularyProcessor.
    :param input_iter:
    :return:
    """
    for row in input_iter:
        all_utterances = []
        context = row[0]
        next_utterances = row[1:101]
        all_utterances.append(context)
        all_utterances.extend(next_utterances)
        for utterance in all_utterances:
            yield utterance

def create_vocab(input_iter, min_frequency):
    """
    Creates and returns a VocabularyProcessor object with the vocabulary
    for the input iterator.
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_sentence_len,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_iter)
    return vocab_processor


def transform_sentence(sequence, vocab_processor):
    """
    Maps a single sentence into the integer vocabulary. Returns a python array.
    """
    return next(vocab_processor.transform([sequence])).tolist()


def create_example_new_format(row, vocab):
    """
    Creates an example as a tensorflow.Example Protocol Buffer object.
    :param row:
    :param vocab:
    :return:
    """
    context = row[0]
    next_utterances = row[1:101]
    target = row[-1]

    context_transformed = transform_sentence(context, vocab)
    context_len = len(next(vocab._tokenizer([context])))

    # New Example
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["target"].int64_list.value.extend([target])

    # Distractor sequences
    for i, utterance in enumerate(next_utterances):
        opt_key = "option_{}".format(i)
        opt_len_key = "option_{}_len".format(i)
        # Utterance Length Feature
        opt_len = len(next(vocab._tokenizer([utterance])))
        example.features.feature[opt_len_key].int64_list.value.extend([opt_len])
        # Distractor Text Feature
        opt_transformed = transform_sentence(utterance, vocab)
        example.features.feature[opt_key].int64_list.value.extend(opt_transformed)
    return example


def create_tfrecords_file(input_filename, output_filename, example_fn):
    """
    Creates a TFRecords file for the given input data and
    example transofmration function
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    print("Creating TFRecords file at {}...".format(output_filename))
    for i, row in enumerate(create_dialog_iter(input_filename)):
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
    """
    Writes the vocabulary to a file, one word per line.
    """
    vocab_size = len(vocab_processor.vocabulary_)
    with open(outfile, "w") as vocabfile:
        for id in range(vocab_size):
            word =  vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + "\n")
    print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
    print("Creating vocabulary...")
    
    p = TRAIN_PATH
    with open(p, 'r') as f:
        train_json = json.load(f)
    dialogs_json = train_json['dialogue_data']
    
    p2 = TRAIN_CANDIDATES_PATH
    with open(p2, 'r') as f:
        candidates_json = json.load(f)
    
    raw_data = combine_data(dialogs_json, candidates_json, 'train')
    with open('raw_data_train.json', 'w') as f:  # writing JSON object
        json.dump(raw_data, f)
    #input_iter = create_dialog_iter(TRAIN_PATH)
    input_iter = create_dialog_iter('raw_data_train.json')
    input_iter = create_utterance_iter(input_iter)
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary.txt file
    write_vocabulary(
        vocab, os.path.join(FLAGS.vocab_path))

    # Save vocab processor
    vocab.save(os.path.join(FLAGS.vocab_processor))

    # Create train.tfrecords
    create_tfrecords_file(
        #input_filename=TRAIN_PATH,
        input_filename= 'raw_data_train.json',
        output_filename=os.path.join(FLAGS.train_out),
        example_fn=functools.partial(create_example_new_format, vocab=vocab))
   
    p3 = VALIDATION_PATH
    with open(p3, 'r') as f:
        valid_json = json.load(f)
    dialogs_json = valid_json['dialogue_data']
    
    p4 = VALIDATION_CANDIDATES_PATH
    with open(p4, 'r') as f:
        candidates_json = json.load(f)
    
    raw_data = combine_data(dialogs_json, candidates_json, 'dev')
    with open('raw_data_dev.json', 'w') as f:  # writing JSON object
        json.dump(raw_data, f)
                                  
                                  
                                  
    # Create validation.tfrecords
    create_tfrecords_file(
        #input_filename=VALIDATION_PATH,
        input_filename='raw_data_dev.json',
        output_filename=os.path.join(FLAGS.validation_out),
        example_fn=functools.partial(create_example_new_format, vocab=vocab))
