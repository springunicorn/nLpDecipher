from django.shortcuts import render
from . import sentiment, classify
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer

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
    # get the input sentence from input box
    data = request.POST.get('inputsentence', False)

    global cls, sent
    # a map from feature indices to feature value(n-grams feature)
    vocab = dict(map(reversed, sent.count_vect.vocabulary_.items()))
    # Tfidf of input sentence
    X = sent.count_vect.transform([data])
    # POSITIVE: 1; NEGATIVE: 0
    prediction = str(cls.predict(X)[0])
    #prediction = 'POSITIVE' if cls.predict(X)[0] == 1 else 'NEGATIVE'

    tmp = X.toarray()[0]
    # model weights from LogisticRegression
    coef = cls.coef_
    # argsort wi*xi
    reasons = np.argsort([tmp[i]*coef[0][i] for i in range(len(tmp))])

    reason = vocab[reasons[-1]] + ' and ' + vocab[reasons[-2]]
    # if prediction == 'POSITIVE':
    #     reason = vocab[reasons[-1]] + ' and ' + vocab[reasons[-2]]
    # else:
    #     reason = vocab[reasons[0]] + ' and ' + vocab[reasons[1]]

    prediction = "The prediction is: " +  prediction
    reason = "The reason is: your input contains words " + reason + \
        ", which has a major impact on the prediction"

    '''
    LIME usage, generate 'test.html' to shwo the resulting graph
    Need to combine the image to our website
    '''
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(sent.count_vect, cls)
    explainer = LimeTextExplainer(class_names=['NEGATIVE','POSITIVE'])
    exp = explainer.explain_instance(data, c.predict_proba, num_features=6)
    exp.save_to_file('test.html')

    data = "Your input sentence is: " + data
    return render(request,'home.html',\
        {'data':data, 'prediction':prediction, 'reason':reason})


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
