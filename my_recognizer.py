import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for n in range(test_set.num_items):
        X, lengths = test_set.get_item_Xlengths(n)
        current_word = None
        current_word_prob = float('-inf')
        wp = {}

        for w in models:
            word = w
            word_model = models[w]

            try:
                word_score = word_model.score(X, lengths)
                wp[word] = word_score
                if word_score > current_word_prob:
                    current_word_prob = word_score
                    current_word = word
            except:
                wp[word] = float('-inf')

        guesses.append(current_word)
        probabilities.append(wp)

    return (probabilities, guesses)
