from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import json
from hibert_model import *

BUFFER_SIZE = 20000
BATCH_SIZE = 16
MIN_LENGTH = 5
EPOCHS = 20
DATA = 'clean_test.json'
checkpoint_path = "./checkpoints/train"
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
seq_length = 20
sum_length = 1
para_length = 10
dropout_rate = 0.1


def readData(filename):
    with open('data/' + filename, 'r') as fp:
        obj = json.load(fp)
        content_batch = []
        summary_batch = []
        for i in range(len(obj)):
            text = obj[i]
            if i % 2 == 0:
                content_batch.append(text)
            else:
                summary_batch.append(text)
    return content_batch, summary_batch



def evaluate(sample_transformer, inp_sentences):
    encoder_input = makeParagraph(inp_sentences, para_length, seq_length)

    encoder_input = tf.convert_to_tensor(encoder_input)
    output = tf.expand_dims([1], 0)
    for i in range(seq_length):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = sample_transformer(encoder_input, output, False)
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, [2]):
            return tf.squeeze(output, axis=0), attention_weights
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights



def main():
    vocab_size = getVocab() 
    input_vocab_size = vocab_size
    target_vocab_size = vocab_size
    content_batch, summary_batch = readData(DATA)
    print("data loaded.")
    
    sample_transformer = Hibert(num_layers, d_model, num_heads, dff, vocab_size, vocab_size, dropout_rate) 
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    getCheckpoint(sample_transformer, optimizer, checkpoint_path)
    print("checkpoint checked.")

    print("start summarizing...")
    with open("gold_standard.json", 'w') as f:
        json.dump(summary_batch, f)
    prediction = []
    for content in content_batch:
        result, attention = evaluate(sample_transformer, content)
        result = [int(i.numpy()) for i in result]
        prediction.append(result)
    with open("prediction.json", 'w') as f:
        print(type(prediction))
        json.dump(prediction, f)
    
    print("Summarization done.")



main()
