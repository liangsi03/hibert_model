from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import random

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates



def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)



def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)



def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
    
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
    
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
    
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])



def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
  
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, combined_mask, dec_padding_mask



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(loss_object, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
  
    return tf.reduce_mean(loss_)


def maskParagraph(inp_sentences, para_length, seq_length):
    masked_sentence = [0 for j in range(seq_length)]
    inp = [[0 for j in range(seq_length)] for i in range(para_length)]
    max_index = min(para_length, len(inp_sentences))
    masked_index = random.randint(1,max_index-1)

    for i in range(para_length):
        if i != masked_index and i < len(inp_sentences):
            for j in range(seq_length):
                if j < len(inp_sentences[i]):
                    inp[i][j] = inp_sentences[i][j]
        elif i == masked_index:
            for j in range(seq_length):
                if j < len(inp_sentences[i]):
                    masked_sentence[j] = inp_sentences[i][j]

    masked_inp = [tf.expand_dims(sentence, 0) for sentence in inp]
    
    return masked_inp, masked_sentence, masked_index



def maskBatch(inp, para_length, seq_length):
    masked_inp_batch = [[] for i in range(para_length)]
    masked_sentence_batch = []
    masked_index_batch = []
    for entry in inp:
        masked_inp, masked_sentence, masked_index = maskParagraph(entry, para_length, seq_length)
        masked_sentence_batch.append(masked_sentence)
        masked_index_batch.append(masked_index)
        for i in range(para_length):
            masked_inp_batch[i].append(masked_inp[i][0])
    masked_inp_batch = tf.convert_to_tensor(masked_inp_batch)
    masked_sentence_batch = tf.convert_to_tensor(masked_sentence_batch)
    masked_index_batch = tf.convert_to_tensor(masked_index_batch)
    masked_inp_batch = tf.cast(masked_inp_batch, tf.int64)
    masked_sentence_batch = tf.cast(masked_sentence_batch, tf.int64)
    masked_index_batch = tf.cast(masked_index_batch, tf.int64)

    return masked_inp_batch, masked_sentence_batch, masked_index_batch


def makeBatch(content, summary, para_length, sum_length, seq_length):
    
    content_batch = [[] for i in range(para_length)]
    summary_batch = []
    
    for i in range(len(content)):
        make_content = makeParagraph(content[i], para_length, seq_length)
        make_summary = makeParagraph(summary[i], sum_length, seq_length)
        for j in range(para_length):
            content_batch[j].append(make_content[j][0])
        summary_batch.append(make_summary[0])

    content_batch = tf.convert_to_tensor(content_batch)
    content_batch = tf.cast(content_batch, tf.int64)
    summary_batch = tf.convert_to_tensor(summary_batch)[:,0,:]
    summary_batch = tf.cast(summary_batch, tf.int64)

    return content_batch, summary_batch



def makeParagraph(inp_sentences, length, seq_length):
    inp = [[0 for j in range(seq_length)] for i in range(length)]

    for i in range(length):
        if i < len(inp_sentences):
            for j in range(seq_length):
                if j < len(inp_sentences[i]):
                    inp[i][j] = inp_sentences[i][j]

    inp = [tf.expand_dims(sentence, 0) for sentence in inp]
    
    return inp



def getVocab():
    with open("data/vocab", "r") as vocab_file:
        lines = [line.strip() for line in vocab_file.readlines()]
        vocab_size = len(lines)
    return vocab_size