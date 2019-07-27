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
    train_dataset = []
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
            if (len(summary_batch) == BATCH_SIZE):
                train_dataset.append((int(i/BATCH_SIZE), content_batch, summary_batch))
                content_batch = []
                summary_batch = []
    return train_dataset



def train_step(sample_transformer, loss_object, optimizer, train_loss, train_accuracy, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = sample_transformer(inp, tar_inp, True)
        loss = loss_function(loss_object, tar_real, predictions)
    
    gradients = tape.gradient(loss, sample_transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, sample_transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)


def main():
    vocab_size = getVocab() 
    input_vocab_size = vocab_size
    target_vocab_size = vocab_size
    train_dataset = readData(DATA)
    print("data loaded.")
    
    sample_transformer = Hibert(num_layers, d_model, num_heads, dff, vocab_size, vocab_size, dropout_rate) 
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    getCheckpoint(sample_transformer, optimizer, checkpoint_path)

    print("start training...")
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
    
        for (batch, content, summary) in train_dataset:
            content, summary = makeBatch(content, summary, para_length, sum_length, seq_length)
            train_step(sample_transformer, loss_object, optimizer, train_loss, train_accuracy, content, summary)

            if batch % 5 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))
    
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

main()