#!/bin/python
import nltk
import string
import numpy as np
from collections import defaultdict
import nltk

vocab = defaultdict(int)

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")

    # tokens = []
    # for x in sentiment.train_data:
    #     if x not in string.punctuation and x not in nltk.corpus.stopwords.words('english'):
    #         tokens.extend(nltk.word_tokenize(x))
    # print(len(tokens))
    # print(len(set(tokens)))

    # sentiment = countvectorizer_feat(sentiment)
    sentiment = tfidfvectorizer_feat(sentiment)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    le_name_mapping = dict(zip(sentiment.le.classes_, sentiment.le.transform(sentiment.le.classes_)))
    print(le_name_mapping)
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def countvectorizer_feat(sentiment):
    from sklearn.feature_extraction.text import CountVectorizer
    import nltk
    # sentiment.count_vect = CountVectorizer(stop_words='english', tokenizer=nltk.word_tokenize)
    sentiment.count_vect = CountVectorizer(tokenizer=nltk.word_tokenize)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    return sentiment

def tfidfvectorizer_feat(sentiment, max_feat=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk

    # from nltk.stem import WordNetLemmatizer 
    # class LemmaTokenizer(object):
    #     def __init__(self):
    #         self.wnl = WordNetLemmatizer()
    #     def __call__(self, articles):
    #         return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(articles)]

    if max_feat != 0:
        sentiment.count_vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,3), tokenizer=nltk.word_tokenize, max_features=max_feat)
    else:
        sentiment.count_vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,3), tokenizer=nltk.word_tokenize)
    
    sentiment.trainX = sentiment.count_vect.fit_transform(init_vocab(sentiment.train_data))
    # sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    return sentiment

def init_vocab(data):
    global vocab
    twd = nltk.tokenize.treebank.TreebankWordDetokenizer()
    new_data = []
    for s in data:
        tmp = nltk.word_tokenize(s)
        new_tmp = []
        for t in tmp:
            n = t if vocab[t] > 1 else 'UNKUNK'
            new_tmp.append(n)
        new_data.append(twd.detokenize(new_tmp))
    return new_data

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    global vocab
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
        if 'train' in fname:
            for w in nltk.word_tokenize(text):
                vocab[w] += 1
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

def save_to_fig(x,y,xaxis,yaxis,title,fname,ticks):
    import matplotlib.pyplot as plt
    plt.plot(x, y, marker='o')
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.title(title)
    plt.xticks(ticks, fontsize='xx-small')
    plt.savefig(fname)


def semi_supervise(sentiment, unlabeled, iter, num_conf):
    import classify
    best_dev = []
    # from scipy.sparse import vstack
    for i in range(iter):
        print("\nTraining classifier")
        sentiment = tfidfvectorizer_feat(sentiment)

        # reference: https://stackoverflow.com/questions/45232671/obtain-tf-idf-weights-of-words-with-sklearn
        index_value = {i[1]:i[0] for i in sentiment.count_vect.vocabulary_.items()}
        fully_indexed = {}
        for row in sentiment.trainX:
            for (column,value) in zip(row.indices,row.data):
                fully_indexed[index_value[column]] = value
        print(sorted(fully_indexed.items(), key=lambda x:x[1], reverse=True)[:10])

        unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 1000)
        acc = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
        if i != 0:
            best_dev.append(acc)
        # preds = cls.predict(unlabeled.X)
        # conf_score = np.max(cls.predict_proba(unlabeled.X), axis=1)

        conf_score = np.apply_along_axis(lambda x: np.random.choice(x,1,p=x)[0], 1, cls.predict_proba(unlabeled.X))
        preds = np.array([int(i >= 0.5) for i in conf_score])

        # conf_score = np.absolute(cls.decision_function(unlabeled.X))
        # conf_idx = np.argsort(conf_score)

        '''
        reference: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        '''
        # def find_nearest(array, value):
        #     array = np.asarray(array)
        #     idx = (np.abs(array - value)).argmin()
        #     return idx

        sum_conf = np.sum(conf_score)
        conf_score = conf_score / sum_conf

        conf_idx = np.random.choice(list(range(len(conf_score))), num_conf, p=conf_score)
        # conf_idx = []
        # for i in conf_tmp:
        #     conf_idx.append(find_nearest(conf_score,i))

        # conf_idx = np.nonzero(conf_score > 0.99)[0]
        # print(len(conf_idx))
        # if len(conf_idx) < 1000:
        #     return unlabeled, cls, sentiment

        # new_labeled_X = np.array(unlabeled.data)[conf_idx[-num_conf:]]
        # new_labeled_y = preds[conf_idx[-num_conf:]]
        new_labeled_X = np.array(unlabeled.data)[conf_idx]
        new_labeled_y = preds[conf_idx]
        tmp_idx = [i for i in range(len(conf_score)) if i not in conf_idx]

        sentiment.train_data = np.concatenate((sentiment.train_data,new_labeled_X))
        sentiment.trainy = np.concatenate((sentiment.trainy,new_labeled_y))
        # unlabeled.data = np.array(unlabeled.data)[conf_idx[:-num_conf]]
        unlabeled.data = np.array(unlabeled.data)[tmp_idx]
    return unlabeled, cls, sentiment, max(best_dev)

# if __name__ == "__main__":
#     import classify
#     print("Reading data")
#     tarfname = "data/sentiment.tar.gz"

#     '''
#     code to tune the maximum number of features to be used in TfidfVectorizer
#     '''
#     # max_feats = [140000, 160000, 170000, 175000, 180000, 200000]
#     # y = []
#     # for feat in max_feats:
#     #     sentiment = read_files(tarfname, feat)
#     #     print("\nTraining classifier")
#     #     cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 1000)
#     #     print("\nEvaluating")
#     #     classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
#     #     y.append(classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev'))
#     # save_to_fig(max_feats,y,'maximum number of features','dev accuracy',\
#     #         'Dev Accuracy vs Maximum Number of Features to be used in Vectorizer',"dev_max_feat.png",max_feats)

#     '''
#     code to tune regularization strength in LogisticRegression
#     '''
#     # reg_stength = [0.1,1,10,50,100,500,800,1000,1100]
#     # dev_score = []
#     # for c in reg_stength:
#     #     cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, c)
#     #     print("\nEvaluating")
#     #     _ = classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
#     #     dev_score.append(classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev'))
#     # ticks = [0.1,50,100,500,800,1000,1100]
#     # save_to_fig(reg_stength,dev_score,'regularization strength','dev accuracy',\
#     #     'Dev Accuracy vs Regularization Strength in LogisticRegression',"dev_C.png",ticks)

#     '''
#     code to produce accuracy report for train, dev, unlabeled data
#     '''
#     # print("\nTraining classifier")

#     # cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 1000)
#     # print("\nEvaluating")
#     # classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
#     # classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

#     print("\nPerforming semi-supervised training")
#     # num_unlabel = [500,1000,2000,3000,5000,10000]
#     num_unlabel = [500]
#     # dev_score = []
#     for n in num_unlabel:
#         print(n)
#         sentiment = read_files(tarfname, 180000)
#         unlabeled = read_unlabeled(tarfname, sentiment)
#         test_X, cls, sentiment, best_dev = semi_supervise(sentiment, unlabeled, 10, n)
#         # dev_score.append(best_dev)
#     # save_to_fig(num_unlabel,dev_score,'number of added unlabeled data','dev accuracy',\
#         # 'Dev Accuracy vs Number of Added unlabeled data',"dev_num_unlabeled.png",num_unlabel)
#     unlabeled = read_unlabeled(tarfname, sentiment)
#     print("Writing predictions to a file")
#     write_pred_kaggle_file(unlabeled, cls, "data/semi-sentiment-pred.csv", sentiment)

#     #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

#     # You can't run this since you do not have the true labels
#     # print "Writing gold file"
#     # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")


def run_script():
    from . import classify

    tarfname = "decipher/data/sentiment.tar.gz"

    sentiment = read_files(tarfname)
    cls1 = classify.train_classifier(sentiment.trainX, sentiment.trainy, 1000, 'l1', 'liblinear', 10000)
    cls2 = classify.train_classifier(sentiment.trainX, sentiment.trainy, 1000, 'l2', 'lbfgs', 10000)
    return cls1, cls2, sentiment

def graph(cls,test_s,X,vocab,fname,title):
    '''
    L1/L2 norm wi*xi graph
    '''
    import matplotlib.pyplot as plt
    L_wts = []
    for i,d in enumerate(cls.coef_[0]):
        if d*test_s[i] != 0:
            L_wts.append((vocab[i],d*test_s[i]))
    if cls.predict(X)[0] == 0:
        sort_L_wts = sorted(L_wts,key=lambda x:x[1])[:8]
    else:
        sort_L_wts = sorted(L_wts,key=lambda x:x[1],reverse=True)[:8]
    cols = []
    for (_,x) in sort_L_wts:
        cols.append('b' if x >= 0 else 'r')
    plt.bar([i for (i,x) in sort_L_wts],[x for (i,x) in sort_L_wts],\
        color=cols)
    plt.xticks(rotation=45)
    plt.xlabel('impactful feature words')
    plt.ylabel('contribution of the word when predicting')
    plt.title(title)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(fname)
    plt.close()
