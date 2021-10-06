import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.layers.preprocessing.text_vectorization import LOWER_AND_STRIP_PUNCTUATION, \
    SPLIT_ON_WHITESPACE


# Text examples

train_texts = [
    "Un accueil, un service, des plats, des vins, des desserts de qualité.",
    "La carte évolue et les vins évoluent.",
    "Les plats du jour sont souvent attractifs et même excellents.",
    "Offre végétarienne tout à fait satisfaisante.",
]

test_texts = [
    "accueil excellent, Très bon accueil.",
    "Service rapide.",
    "Pizza excellente car la pâte est fine et croustillante.",
    "Tiramitsu fait maison.",
]

########
vocab_size = 12
max_seq_length = 15
emb_dim = 10
lstm_hidden_dim = 8


vectorizer = TextVectorization(
    standardize=LOWER_AND_STRIP_PUNCTUATION,
    split=SPLIT_ON_WHITESPACE,
    ngrams=(1,1),
    max_tokens=vocab_size,  # max size of the vocabulary, use max_tokens=None for unlimited size
    output_mode='int',  # encode each token to an int index (the indices of the tokens in the vocabulary)
    output_sequence_length=max_seq_length,  # sequence length to pad the output to
)

vectorizer.adapt(train_texts)

# Model arch: 1 intermediary layer
# input
input = tf.keras.Input(shape=(max_seq_length,), name='input')
#
# Embedding layer with vocabulary size and embedding dimension
X = tf.keras.layers.Embedding(vocab_size, emb_dim)(input)
# BiLSTM to encode each text
encoded_texts = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_hidden_dim))(X)
# or you can also return the contextual vector of each token
#encoded_texts = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_hidden_dim, return_sequences=True))(X)

# define model with input and output
encoder_model = tf.keras.Model(inputs=[input], outputs=[encoded_texts])

encoder_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
encoder_model.summary()

# test

input_ids = vectorizer(test_texts)
output = encoder_model(input_ids)
print(output.shape)
print(output)






