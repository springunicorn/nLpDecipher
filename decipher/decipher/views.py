from django.shortcuts import render
from . import sentiment, classify
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import nltk
import matplotlib.pyplot as plt
import random

cls1 = None
cls2 = None
cls3 = None
cls4 = None
sent = None
sent_gender = None


def train(request):
    global cls1, cls2, cls3, cls4, sent, sent_gender
    cls1, cls2, sent = sentiment.run_script("decipher/data/sentiment.tar.gz")
    cls3, cls4, sent_gender = sentiment.run_script(
        "decipher/data/gender.tar.gz", c=1)
    progress = 'Training Done'
    return render(request, 'home.html', {'progress': progress})


def button(request):
    return render(request, 'home.html')


def output(request):
    global cls1, cls2, sent
    twd = nltk.tokenize.treebank.TreebankWordDetokenizer()
    # get the input sentence from input box
    data = request.POST.get('inputsentence', False).lower()
    # deal with rare words
    tmp_data = nltk.word_tokenize(data)
    rare_words = []
    tmp = []
    for x in tmp_data:
        if x not in sent.count_vect.vocabulary_:
            choice = random.random()
            if choice < 0.5:
                tmp.append('unkunk')
                rare_words.append(x)
            else:
                tmp.append(x)
        else:
            tmp.append(x)
    rdata = twd.detokenize(tmp)

    # a map from feature indices to feature value(n-grams feature)
    vocab = dict(map(reversed, sent.count_vect.vocabulary_.items()))
    # Tfidf of input sentence
    X = sent.count_vect.transform([rdata])
    # POSITIVE: 1; NEGATIVE: 0
    prediction = 'POSITIVE' if cls2.predict(X)[0] == 1 else 'NEGATIVE'

    test_s = X.toarray()[0]

    # L1-norm weights
    sentiment.graph(cls1, test_s, X, vocab, 'static/decipher/l1.jpg', 'L1-norm weights')

    # model weights from LogisticRegression
    coef = cls2.coef_
    # wi*xi
    wixi = [test_s[i]*coef[0][i] for i in range(len(test_s))]
    # argsort wi*xi
    reasons = np.argsort(wixi)

    # L2-norm weights
    sentiment.graph(cls2, test_s, X, vocab, 'static/decipher/l2.jpg', 'L2-norm weights')

    if prediction == 'POSITIVE':
        if wixi[reasons[-2]] != 0:
            reason = vocab[reasons[-1]] + ' and ' + vocab[reasons[-2]]
        else:
            reason = vocab[reasons[-1]]
    else:
        if wixi[reasons[1]] != 0:
            reason = vocab[reasons[0]] + ' and ' + vocab[reasons[1]]
        else:
            reason = vocab[reasons[0]]

    prediction = "The prediction is: " + prediction
    reason = "The reason is: your input contains words " + reason + \
        ", which have a major impacts on the prediction"

    if 'unkunk' in reason:
        confusion = "Since the input sentence contains \
            a list of rare words {}, we are not confident to give this\
                prediction".format(', '.join(rare_words))
    else:
        confusion = None

    '''
    LIME usage, generate 'test.html' to shwo the resulting graph
    Need to combine the image to our website
    '''
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(sent.count_vect, cls2)
    explainer = LimeTextExplainer(class_names=['NEGATIVE', 'POSITIVE'])
    exp = explainer.explain_instance(rdata, c.predict_proba, num_features=10)
    exp.save_to_file('static/decipher/lime.html')

    data = "Your input sentence is: " + data
    return render(request, 'home.html',
                  {'data': data, 'prediction': prediction, 'reason': reason, 'confusion': confusion})


def output2(request):
    global cls3, cls4, sent_gender

    twd = nltk.tokenize.treebank.TreebankWordDetokenizer()
    # get the input sentence from input box
    data = request.POST.get('inputsentence2', False).lower()
    # deal with rare words
    tmp_data = nltk.word_tokenize(data)
    rare_words = []
    tmp = []
    for x in tmp_data:
        if x not in sent_gender.count_vect.vocabulary_:
            choice = random.random()
            if choice < 0.5:
                tmp.append('unkunk')
                rare_words.append(x)
            else:
                tmp.append(x)
        else:
            tmp.append(x)
    rdata = twd.detokenize(tmp)

    # a map from feature indices to feature value(n-grams feature)
    vocab = dict(map(reversed, sent_gender.count_vect.vocabulary_.items()))
    # Tfidf of input sentence
    X = sent_gender.count_vect.transform([rdata])

    predidx = cls4.predict(X)[0]
    classmap = ['brand', 'female', 'male']
    prediction = classmap[predidx]

    tmp = X.toarray()[0]

    sentiment.gender_graph(cls3, tmp, X, vocab, 'static/decipher/l1.jpg', 'L1-norm weights')

    # model weights from LogisticRegression
    coef = cls4.coef_
    # wi*xi
    wixi = [tmp[i]*coef[predidx][i] for i in range(len(tmp))]
    # argsort wi*xi
    reasons = np.argsort(wixi)

    sentiment.gender_graph(cls4, tmp, X, vocab, 'static/decipher/l2.jpg', 'L2-norm weights')

    if wixi[reasons[-2]] != 0:
        reason = vocab[reasons[-1]] + ' and ' + vocab[reasons[-2]]
    else:
        reason = vocab[reasons[-1]]

    prediction = "The prediction is: " + prediction
    reason = "The reason is: your input contains words " + reason + \
        ", which has a major impact on the prediction"

    if 'unkunk' in reason:
        confusion = "Since the input sentence contains \
            a list of rare words {}, we are not confident to give this\
                prediction".format(', '.join(rare_words))
    else:
        confusion = None

    '''
    LIME usage, generate 'test.html' to shwo the resulting graph
    Need to combine the image to our website
    '''
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(sent_gender.count_vect, cls4)
    explainer = LimeTextExplainer(class_names=['brand', 'female', 'male'])
    exp = explainer.explain_instance(
        rdata, c.predict_proba, num_features=10, top_labels=3)
    exp.save_to_file('static/decipher/lime.html')

    data = "Your input sentence is: " + data
    return render(request, 'home.html',
                  {'data': data, 'prediction': prediction, 'reason': reason})
