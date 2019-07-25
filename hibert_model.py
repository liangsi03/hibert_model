from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from encoder import *
from decoder import *

class Hibert(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
        super(Hibert, self).__init__()

        self.sentence_encoder = SentenceEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.paragraph_encoder = ParagraphEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training):
        sentence_size = inp.shape[0]
        batch_size = inp.shape[1]
        d_model = inp.shape[2]

        sentence_encoder_output = [[] for i in range(batch_size)]

        for i in range(len(inp)):
            sentence = inp[i]
            enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(sentence, tar)

            next_sentence_encoder_output = self.sentence_encoder(sentence, training, enc_padding_mask)
            for batch in range(batch_size):
                sentence_encoder_output[batch].append(next_sentence_encoder_output[batch])
                
        sentence_encoder_output = tf.convert_to_tensor(sentence_encoder_output)
        
        paragraph_encoder_output = self.paragraph_encoder(sentence_encoder_output, training, None)
        
        score = tf.nn.softmax(paragraph_encoder_output, axis=1)
        score = tf.reduce_sum(score, 2)
        max_score = tf.reduce_max(score, 1)
        max_index = []
        for i in range(batch_size):
            for j in range(sentence_size):
                if tf.math.equal(score[i][j], max_score[i]):
                    max_index.append(j)
                    break
        
        encoder_output = [paragraph_encoder_output[i][max_index[i]] for i in range(batch_size)]
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        
        dec_output, attention_weights = self.decoder(tar, encoder_output, 
                                                     training, look_ahead_mask, dec_padding_mask)
    
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
        return final_output, attention_weights