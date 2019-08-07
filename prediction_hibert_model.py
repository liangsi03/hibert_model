from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from encoder import *
from prediction_decoder import *

class PredictionHibert(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
        super(PredictionHibert, self).__init__()

        self.sentence_encoder = SentenceEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.paragraph_encoder = ParagraphEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = PredictionDecoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)    
    
    def call(self, inp, tar, masked_index, training):
        # inp == (sequence_length, batch_size, d_model)
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

        masked_encoder_output = [paragraph_encoder_output[i][masked_index[i]] for i in range(batch_size)]
        # dec_output.shape == (batch_size, tar_seq_len, d_model)

        dec_output, attention_weights = self.decoder(tar, masked_encoder_output, 
                                                     training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights
