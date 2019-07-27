import json
import tensorflow as tf
from prediction_hibert_model import *

BUFFER_SIZE = 20000
BATCH_SIZE = 16
MIN_LENGTH = 5
EPOCHS = 20
DATA = '.json'
checkpoint_path = "./checkpoints/train"
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
seq_length = 20
para_length = 10
dropout_rate = 0.1
filename = ".json"



def evaluate(sample_transformer, inp_sentences):
    masked_sentence = [0 for j in range(seq_length)]
    inp = [[0. for j in range(seq_length)] for i in range(len(inp_sentences))]
    masked_index = random.randint(1,len(inp_sentences)-1)
    
    for i in range(len(inp_sentences)):
        if i != masked_index:
            for j in range(seq_length):
                if j < len(inp_sentences[i]):
                    inp[i][j] = float(inp_sentences[i][j])
        else:
            for j in range(seq_length):
                if j < len(inp_sentences[i]):
                    masked_sentence[j] = float(inp_sentences[i][j])

    encoder_input = tf.convert_to_tensor([tf.expand_dims(sentence, 0) for sentence in inp])
    output = tf.expand_dims([1], 0)
    masked_index = tf.convert_to_tensor([masked_index])
    
    for i in range(seq_length):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = sample_transformer(encoder_input, output, masked_index, False)
        # select the last word from the seq_len dimension 
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, [2]):
            return tf.squeeze(output, axis=0), attention_weights, inp_sentences[masked_index[0]]
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0), attention_weights, inp_sentences[masked_index[0]]



def readData(filename):
    train_dataset = []
    with open('data/' + filename, 'r') as fp:
        obj = json.load(fp)
        for i in range(len(obj)):
            if (len(obj[i]) > MIN_LENGTH):
                train_dataset.append(obj[i])
    return test_dataset



def translate():
    vocab_size = getVocab() 
    input_vocab_size = vocab_size
    target_vocab_size = vocab_size
    test_dataset = readData(filename)
    print("data loaded.")

    sample_transformer = PredictionHibert(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)    
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    getCheckpoint(sample_transformer, optimizer, checkpoint_path)
    print("checkpoint checked.")

    print("start predicting...")
    prediction = []
    gold_standard = []
    for sentence in test_dataset:
        result, attention, gold = evaluate(sample_transformer, sentence)
        result = [int(i.numpy()) for i in result]
        prediction.append(result)
        gold_standard.append(gold)
    with open("data/prediction_result.json", 'w') as f:
        json.dump(prediction, f)
    with open("data/prediction_gold.json", 'w') as f:
        json.dump(gold_standard, f)
    
    print("Prediction finished.")

translate()