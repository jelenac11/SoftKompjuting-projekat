import numpy as np
import string
from string import digits
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from emot.emo_unicode import EMOTICONS
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from torchtext import data
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.cost import cross_entropy_seq_with_mask
from tensorlayer.models import Seq2seq
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
stopwords.words('english')
from spellchecker import SpellChecker

chat_words = {
    "afaik": "as far as i know",
    "asap": "as soon as possible",
    "atk": "at the keyboard",
    "atm": "at the moment",
    "brb": "be right back",
    "brt": "be right there",
    "btw": "by the way",
    "b4": "before",
    "cu": "see you",
    "cya": "see you",
    "faq": "frequently asked questions",
    "fc": "fingers crossed",
    "fyi": "for your information",
    "gn": "good night",
    "gr8": "great",
    "g9": "genius",
    "ic": "i see",
    "imo": "in my opinion",
    "iow": "in other words",
    "lol": "laughing out loud",
    "l8r": "later",
    "mte": "my thoughts rxactly",
    "m8": "mate",
    "nrn": "no reply necessary",
    "oic": "oh i see",
    "rofl": "rolling on the floor laughing",
    "thx": "thank you",
    "ttyl": "talk to you later",
    "u": "you",
    "u2": "you too",
    "w8": "wait",
    "imma": "i am going to",
    "2nite": "tonight"
}

contractions = {
    "ain't": "are not",
    "aren't": "are not",
    "'bout": "about",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "here's": "here is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "i phone": "iphone",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "'til": "until",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


def remove_chat_words_and_contractions(text):
    new_text = []
    for word in text.split(' '):
        if word in chat_words.keys():
            new_text += chat_words[word].split(' ')
        if word in contractions.keys():
            new_text += contractions[word].split(' ')
        else:
            new_text.append(word)
    return ' '.join(new_text)


def remove_urls(text):
    return re.sub(r'https://t\.co/\w+', 'url', text)


def remove_mentions(text):
    return re.sub(r'@[0-9A-Za-z_\-]+', '', text)


def remove_hashtags(text):
    return re.sub(r'#[0-9A-Za-z_\-]+', '', text)


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)


puncList = ["&gt","amp","%",'newlinechar',"~",".","\t","\n", "^", "_", "*", "<", ">", ";", ":", "!", "?", "/", "\\", ",", "#", "@", "$", "&", ")", "(", "\"", "]", "[", "|", "{", "}","=","-","+","\""]
isascii = lambda s: len(s) == len(s.encode())
remove_digits = str.maketrans('', '', digits)


def remove_numbers(content):
    return content.translate(remove_digits)


def remove_inerpunction_and_nonascii_chars(content):
    new_content = ""
    content = content.split(" ")
    for word in content:
        word = word.strip()
        for punc in puncList:
            word = ''.join(word.split(punc))
        if word != "" and word != " " and isascii(word):
            new_content = new_content + " " + word
    return new_content.strip()


def remove_spacing(text):
    return " ".join([word for word in nltk.word_tokenize(text)])


def clean(text):
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"’", "'", text)
    text = text.lower()
    text = remove_chat_words_and_contractions(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&lt;', '<', text)
    text = remove_emojis(text)
    text = remove_emoticons(text)
    text = remove_inerpunction_and_nonascii_chars(text)
    text = remove_numbers(text)
    text = remove_spacing(text)
    return text
