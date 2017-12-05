import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        min_score = float("inf")
        min_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                score = model.score(self.X, self.lengths)

                n_features = len(self.X[0])
                transition_numbers = n * (n - 1)

                n_params = transition_numbers + 2 * n_features * n

                logN = np.log(len(self.X))

                current_bic = -2 * score + n_params * logN
                if current_bic < min_score:
                    min_score = current_bic
                    min_model = model

            except:
                continue

        if min_model is None:
            return self.base_model(self.n_constant)

        return min_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    result_dict = None

    def prepare(self):
        if SelectorDIC.result_dict is not None:
            return SelectorDIC.result_dict
        result_dict = {}
        for n in range(self.min_n_components, self.max_n_components + 1):
            for w in self.words:
                X, lengths = self.hwords[w]

                try:
                    word_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                             random_state=self.random_state, verbose=False).fit(X, lengths)
                    word_score = word_model.score(X, lengths)
                    result_dict[(w, n)] = (word_model, word_score)
                except:
                    result_dict[(w, n)] = (None, float("-inf"))

                SelectorDIC.result_dict = result_dict
        return result_dict

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_dic = float("-inf")
        max_model = None

        dict = self.prepare()

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                current_word_key = (self.this_word, n)
                current_word_score = dict[current_word_key][1]

                others = [w for w in self.words if w is not self.this_word]
                dic = current_word_score - np.mean([dict[(w, n)][1] for w in others if dict[(w, n)][0] is not None])

                if dic > max_dic:
                    max_dic = dic
                    max_model = dict[current_word_key][0]
            except:
                continue

        if max_model is None:
            return self.base_model(self.n_constant)

        return max_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)



        max_cv = float("-inf")
        max_model = None
        spl = 3

        for n in range(self.min_n_components, self.max_n_components + 1):
            res = []

            if len(self.sequences) < spl:
                break

            split_method = KFold(spl)

            for train, test in split_method.split(self.sequences):
                X_test, lengths_test = combine_sequences(test, self.sequences)
                X_train, lengths_train = combine_sequences(train, self.sequences)

                try:
                    word_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                             random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                    word_score = word_model.score(X_test, lengths_test)
                    res.append(word_score)
                except:
                    continue

            if len(res) > 0:
                avg = np.average(res)

                if avg > max_cv:
                    max_cv = avg
                    max_model = word_model

        if max_model is None:
            return self.base_model(self.n_constant)

        return max_model
