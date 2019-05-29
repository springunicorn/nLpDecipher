from django.shortcuts import render
from . import sentiment, classify
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import nltk

cls = None
sent = None

def train(request):
    global cls, sent
    cls, sent = sentiment.run_script()
    progress = 'Training Done'
    return render(request,'home.html',{'progress':progress})

def button(request):
    return render(request,'home.html')

def output(request):
    global cls, sent
    twd = nltk.tokenize.treebank.TreebankWordDetokenizer()
    # get the input sentence from input box
    data = request.POST.get('inputsentence', False).lower()
    # deal with rare words
    tmp_data = nltk.word_tokenize(data)
    rare_words = []
    tmp = []
    for x in tmp_data:
        if x not in sent.count_vect.vocabulary_:
            tmp.append('unkunk')
            rare_words.append(x)
        else:
            tmp.append(x)
    rdata = twd.detokenize(tmp)

    # a map from feature indices to feature value(n-grams feature)
    vocab = dict(map(reversed, sent.count_vect.vocabulary_.items()))
    # Tfidf of input sentence
    X = sent.count_vect.transform([rdata])
    # POSITIVE: 1; NEGATIVE: 0
    prediction = 'POSITIVE' if cls.predict(X)[0] == 1 else 'NEGATIVE'

    test_s = X.toarray()[0]
    # model weights from LogisticRegression
    coef = cls.coef_
    # wi*xi
    wixi = [test_s[i]*coef[0][i] for i in range(len(test_s))]
    # argsort wi*xi
    reasons = np.argsort(wixi)

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

    prediction = "The prediction is: " +  prediction
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
    c = make_pipeline(sent.count_vect, cls)
    explainer = LimeTextExplainer(class_names=['NEGATIVE','POSITIVE'])
    exp = explainer.explain_instance(rdata, c.predict_proba, num_features=len(rdata.split(' ')))
    exp.save_to_file('test.html')

    data = "Your input sentence is: " + data
    return render(request,'home.html',\
                  {'data': data, 'prediction': prediction, 'reason': reason, 'confusion': confusion})


def output2(request):
    # get the input sentence from input box
    data = request.POST.get('inputsentence2', False)

    global cls, sent
    # a map from feature indices to feature value(n-grams feature)
    vocab = dict(map(reversed, sent.count_vect.vocabulary_.items()))
    # Tfidf of input sentence
    X = sent.count_vect.transform([data])

    predidx = cls.predict(X)[0]
    classmap = ['brand', 'female', 'male']
    prediction = classmap[predidx]

    tmp = X.toarray()[0]
    # model weights from LogisticRegression
    coef = cls.coef_
    # argsort wi*xi
    reasons = np.argsort([tmp[i]*coef[0][i] for i in range(len(tmp))])

    reason = vocab[reasons[-1]] + ' and ' + vocab[reasons[-2]]

    prediction = "The prediction is: " + prediction
    reason = "The reason is: your input contains words " + reason + \
        ", which has a major impact on the prediction"

    '''
    LIME usage, generate 'test.html' to shwo the resulting graph
    Need to combine the image to our website
    '''
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(sent.count_vect, cls)
    explainer = LimeTextExplainer(class_names=['brand', 'female', 'male'])
    exp = explainer.explain_instance(data, c.predict_proba, num_features=6, top_labels=3)
    exp.save_to_file('test.html')

    data = "Your input sentence is: " + data
    return render(request, 'home.html',
                  {'data': data, 'prediction': prediction, 'reason': reason})
    
