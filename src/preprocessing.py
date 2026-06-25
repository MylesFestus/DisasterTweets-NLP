import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from nltk import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()      # Convert to lowercase
    text = re.sub(r'\W', ' ', text)       # Replace non-word characters with spaces

    words = text.split()       # Split into words

    words = [w.translate(str.maketrans('', '', string.punctuation)) for w in words]    # Remove punctuation

    words = [w for w in words if w.isalpha() and len(w)>1]           # Keep only alphabetic words

    words = [w for w in words if w not in stop_words]           # Remove stop words

    return words


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def tokenize(list_of_words):

  tagged_tokens = pos_tag(list_of_words)

  words = [
      lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag))
      for word, tag in tagged_tokens]

  return " ".join(words)

