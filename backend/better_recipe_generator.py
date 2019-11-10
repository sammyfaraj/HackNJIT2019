import tensorflow as tf
from sklearn.model_selection import train_test_split

import configparser
import unicodedata
import re
import numpy as np
import os
import io
import time
import pickle
from random import randint

class Encoder(tf.keras.Model):
    def __init__(self, inp_lang, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.inp_lang   = inp_lang
        self.batch_sz   = batch_sz
        self.enc_units  = enc_units
        self.embedding  = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self, batch=None):
        if not batch: batch=self.batch_sz
        return tf.zeros((batch, self.enc_units))

    
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V  = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        score = self.V(tf.nn.tanh( self.W1(values) + self.W2(hidden_with_time_axis) ))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, targ_lang, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.targ_lang  = targ_lang
        self.dec_units  = dec_units
        self.embedding  = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru_1 = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.gru_2 = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru_1(x, initial_state=hidden)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights


class Generator():
    def __init__(self, path, config=None, lang=None):
        self.path = path

        if not lang:    inp_lang, targ_lang = self.load_lang()
        else:           inp_lang, targ_lang = lang

        if not config:  self.config = self.load_config()
        else:           self.config = config


        self.encoder = Encoder(inp_lang, self.config['vocab_inp_size'], self.config['embedding_dim'], self.config['hidden_embed'], self.config['BATCH_SIZE'])
        self.decoder = Decoder(targ_lang, self.config['vocab_tar_size'], self.config['embedding_dim'], self.config['hidden_embed'])

        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True, reduction='none' )

        self.checkpoint_prefix = os.path.join(path, "ckpt")
        self.checkpoint = tf.train.Checkpoint(  optimizer=self.optimizer,
                                                encoder=self.encoder,
                                                decoder=self.decoder)
        if not config: self.load_weights()
        

    def predict(self, sentence, temperature=None):

        sentence = self.preprocess_sentence(sentence)

        inputs = [self.encoder.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=self.config['max_length_inp'],
                                                            padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = self.encoder.initialize_hidden_state(1)
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.decoder.targ_lang.word_index['<start>']], 0)

        for t in range(self.config['max_length_targ']):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)

            attention_weights = tf.reshape(attention_weights, (-1, ))

            if temperature:
                predictions = predictions / temperature
                predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            else:
                predicted_id = tf.argmax(predictions[0]).numpy()

            if self.decoder.targ_lang.index_word[predicted_id] == '<end>':
                return result
            
            result += self.decoder.targ_lang.index_word[predicted_id] + ' '

            dec_input = tf.expand_dims([predicted_id], 0)

        return result


    # @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.decoder.targ_lang.word_index['<start>']] * self.config['BATCH_SIZE'], 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)
                
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    def train(self, EPOCHS, steps_per_epoch, dataset):
        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                batch,
                                                                batch_loss.numpy()))
            if (epoch + 1) % 2 == 0:
                self.save()

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    def preprocess_sentence(self, w):
        w = ''.join(c for c in unicodedata.normalize('NFD', w.lower().strip()) if unicodedata.category(c) != 'Mn')
        w = unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([0-9.,/])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        w = re.sub(r"[^a-zA-Z0-9.,/]+", " ", w)

        w = w.rstrip().strip()

        w = '<start> ' + w + ' <end>'
        return w


    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def save(self):
        with open(self.path+'config.pkl', 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.path+'lang.pkl', 'wb') as handle:
            pickle.dump((self.encoder.inp_lang,self.decoder.targ_lang), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        

    def load_lang(self): 
        with open(self.path+'lang.pkl', 'rb') as handle:
            lang = pickle.load(handle)
        return lang

    def load_config(self):
        with open(self.path+'config.pkl', 'rb') as handle:
            config = pickle.load(handle)
        return config

    def load_weights(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.path))

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([0-9.,/])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = re.sub(r"[^a-zA-Z0-9.,/]+", " ", w)

    w = w.rstrip().strip()

    w = '<start> ' + w + ' <end>'
    return w


def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang, top_k):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                            oov_token="<unk>",
                                                            filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                            padding='post')

    return tensor, lang_tokenizer

def load_dataset(inpt, targ, test_size=0.5, num_examples=-1):

    inpt_targ = [[preprocess_sentence(i),preprocess_sentence(t)] for i, t in zip(inpt, targ) if len(preprocess_sentence(t).split(' ')) <= config['max_seq']]
    inpt, targ = zip(*inpt_targ[:num_examples])

    input_tensor, inp_lang_tokenizer = tokenize(inpt, 5000)
    target_tensor, targ_lang_tokenizer = tokenize(targ, 5000)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
    
def setup_data(config, inpt, targ):
    BUFFER_SIZE     = 10000
    num_examples    = -1

    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(inpt, targ, config['max_seq'], num_examples)
    config['max_length_targ'], config['max_length_inp'] = max_length(target_tensor), max_length(input_tensor)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.05)
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    steps_per_epoch = len(input_tensor_train)//config['BATCH_SIZE']
    config['vocab_inp_size'] = len(inp_lang.word_index)+1 if len(inp_lang.word_index)+1 < inp_lang.num_words else inp_lang.num_words
    config['vocab_tar_size'] = len(targ_lang.word_index)+1 if len(targ_lang.word_index)+1 < targ_lang.num_words else targ_lang.num_words

    print("Ingredients:",config['vocab_inp_size'])
    print("Instructions:",config['vocab_tar_size'])

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(config['BATCH_SIZE'], drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))
    print(example_input_batch.shape, example_target_batch.shape)

    return config, dataset, steps_per_epoch, (inp_lang, targ_lang), (input_tensor_val, target_tensor_val)


if __name__ == "__main__":
    pass