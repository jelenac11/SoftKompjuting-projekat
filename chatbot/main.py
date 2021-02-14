import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Seq2seq
from preprocessing import clean
from flask import Flask, render_template, request
import pickle

BATCH_SIZE = 32
EPOCHS_NUMBER = 40
VOCABULARY_SIZE_APPLE = 20000
VOCABULARY_SIZE_AMAZON = 23000
VOCABULARY_SIZE_UBER = 10000
EMBEDDING_SIZE = 1024
DECODER_SEQ_LENGTH_APPLE = 25
DECODER_SEQ_LENGTH_AMAZON = 20
DECODER_SEQ_LENGTH_UBER = 21
N_LAYER = 3
N_UNITS = 256
LEARNING_RATE = 0.001

model_apple = Seq2seq(
    decoder_seq_length=DECODER_SEQ_LENGTH_APPLE,
    cell_enc=tf.keras.layers.LSTMCell,
    cell_dec=tf.keras.layers.LSTMCell,
    n_layer=N_LAYER,
    n_units=N_UNITS,
    embedding_layer=tl.layers.Embedding(vocabulary_size=VOCABULARY_SIZE_APPLE + 3, embedding_size=EMBEDDING_SIZE),
)

model_amazon = Seq2seq(
    decoder_seq_length=DECODER_SEQ_LENGTH_AMAZON,
    cell_enc=tf.keras.layers.LSTMCell,
    cell_dec=tf.keras.layers.LSTMCell,
    n_layer=N_LAYER,
    n_units=N_UNITS,
    embedding_layer=tl.layers.Embedding(vocabulary_size=VOCABULARY_SIZE_AMAZON + 3, embedding_size=EMBEDDING_SIZE),
)

model_uber = Seq2seq(
    decoder_seq_length=DECODER_SEQ_LENGTH_UBER,
    cell_enc=tf.keras.layers.LSTMCell,
    cell_dec=tf.keras.layers.LSTMCell,
    n_layer=N_LAYER,
    n_units=N_UNITS,
    embedding_layer=tl.layers.Embedding(vocabulary_size=VOCABULARY_SIZE_UBER + 3, embedding_size=EMBEDDING_SIZE),
)


def predict_apple(query):
    with open('pretrained-models/apple/vocabAppleSupport.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('pretrained-models/apple/invertedVocabAppleSupport.pkl', 'rb') as f:
        inverted_vocab = pickle.load(f)
    model_apple.eval()
    query = clean(query)
    new_query = query
    for el in query.split():
        if el not in inverted_vocab.keys():
            new_query = ''.join(new_query.split(el))
    query = new_query
    query = ' '.join(query.split())
    if query != "":
        query_tokenized = [inverted_vocab.get(w) for w in query.split()]
        answer_tokenized = model_apple(inputs=[[query_tokenized]], seq_length=DECODER_SEQ_LENGTH_APPLE, start_token=VOCABULARY_SIZE_APPLE + 1, top_n=1)
        answer = []
        for word_token in answer_tokenized[0]:
            w = vocab[word_token.numpy()]
            if w == '<end>':
                break
            answer = answer + [w]
        return " ".join(answer)
    return ""


def predict_amazon(query):
    with open('pretrained-models/amazon/vocabAmazon.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('pretrained-models/amazon/invertedVocabAmazon.pkl', 'rb') as f:
        inverted_vocab = pickle.load(f)
    print(len(vocab))
    model_amazon.eval()
    query = clean(query)
    new_query = query
    for el in query.split():
        if el not in inverted_vocab.keys():
            new_query = ''.join(new_query.split(el))
    query = new_query
    query = ' '.join(query.split())
    if query != "":
        query_tokenized = [inverted_vocab.get(w) for w in query.split()]
        answer_tokenized = model_amazon(inputs=[[query_tokenized]], seq_length=DECODER_SEQ_LENGTH_AMAZON, start_token=VOCABULARY_SIZE_AMAZON + 1, top_n=1)
        answer = []
        for word_token in answer_tokenized[0]:
            w = vocab[word_token.numpy()]
            if w == '<end>':
                break
            answer = answer + [w]
        return " ".join(answer)
    return ""


def predict_uber(query):
    with open('pretrained-models/uber/vocabUber.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('pretrained-models/uber/invertedVocabUber.pkl', 'rb') as f:
        inverted_vocab = pickle.load(f)
    model_uber.eval()
    query = clean(query)
    new_query = query
    for el in query.split():
        if el not in inverted_vocab.keys():
            new_query = ''.join(new_query.split(el))
    query = new_query
    query = ' '.join(query.split())
    if query != "":
        query_tokenized = [inverted_vocab.get(w) for w in query.split()]
        answer_tokenized = model_uber(inputs=[[query_tokenized]], seq_length=DECODER_SEQ_LENGTH_UBER, start_token=VOCABULARY_SIZE_UBER + 1, top_n=1)
        answer = []
        for word_token in answer_tokenized[0]:
            w = vocab[word_token.numpy()]
            if w == '<end>':
                break
            answer = answer + [w]
            if "thank" in query:
                possible = ["you are welcome", "happy to help"]
                return

        return " ".join(answer)
    return ""


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get-apple")
def get_bot_response_apple():
    query = request.args.get('msg')
    return predict_apple(query)


@app.route("/get-amazon")
def get_bot_response_amazon():
    query = request.args.get('msg')
    return predict_amazon(query)


@app.route("/get-uber")
def get_bot_response_uber():
    query = request.args.get('msg')
    return predict_uber(query)


if __name__ == '__main__':
    weightsApple = tl.files.load_npz(name='pretrained-models/apple/modelAppleSupport.npz')
    tl.files.assign_weights(weightsApple, model_apple)

    weightsAmazon = tl.files.load_npz(name='pretrained-models/amazon/modelAmazonHelp.npz')
    tl.files.assign_weights(weightsAmazon, model_amazon)

    weightsUber = tl.files.load_npz(name='pretrained-models/uber/modelUber_Support.npz')
    tl.files.assign_weights(weightsUber, model_uber)

    app.run()
