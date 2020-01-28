import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
stopWords = set(stopwords.words('english'))

MAX_NUM_WORD = 500
EMBEDDING_SIZE = 100
CLASSES = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
num_classes = len(CLASSES)
class_dict = dict(zip(CLASSES, range(num_classes)))

def read_vector(filename):
    wordVectors = []
    vocab = []
    fileObject = open(filename, 'r')
    for i, line in enumerate(fileObject):
        if i==0 or i==1: # first line is a number (vocab size)
            continue
        line = line.strip()
        word = line.split()[0]
        vocab.append(word)
        wv_i = []
        for j, vecVal in enumerate(line.split()[1:]):
            wv_i.append(float(vecVal))
        wordVectors.append(wv_i)
    wordVectors = np.asarray(wordVectors)
    vocab_dict = dict(zip(vocab, range(1, len(vocab)+1))) # no 0 id; saved for padding
    print("Vectors read from: "+filename)
    return wordVectors, vocab_dict, vocab

def read_rnn_data(path, vocab_dict):
    myFile= open(path, "rU")
    labels = []
    docIDs = []
    for i, aRow in enumerate(myFile):
        x = [text.split(',') for text in aRow.split()]
        ids = []
        for w in x[1]:
            if w in vocab_dict:
                ids.append(vocab_dict[w])
            else:
                ids.append(vocab_dict["[UNK]"])
        if len(ids)>=5:
            labels.append(class_dict[x[0][0]])
            docIDs.append(ids)
    myFile.close()
    num_docs = len(labels)
    print(num_docs, "docs in total")
    y = np.zeros((num_docs, num_classes), dtype=np.int32)
    x = np.zeros((num_docs, MAX_NUM_WORD), dtype=np.int32)
    for i in range(num_docs):
        y[i][labels[i]] = 1
        if len(docIDs[i])>MAX_NUM_WORD:
            x[i, :] = docIDs[i][:MAX_NUM_WORD]
        else:
            x[i, :len(docIDs[i])] = docIDs[i]
    return x, y, num_docs

def decrease_vocab(path, vocab, target_vocab_size=2000):
    _vocab = [w for w in vocab if w not in stopWords]
    myFile= open(path, "rU")
    docs = []
    for i, aRow in enumerate(myFile):
        x = aRow.split()[1]
        docs.append(x)
    myFile.close()
    vectorizer = TfidfVectorizer(vocabulary = _vocab)
    X = vectorizer.fit_transform(docs)
    word_importance = np.sum(X, axis = 0) # shape: [1, vocab_size], a numpy matrix!
    sorted_vocab_idx = np.squeeze(np.asarray(np.argsort(word_importance), dtype=np.int32)) # shape: [vocab_size, ], a numpy array
    vocab_idx_wanted = np.flip(sorted_vocab_idx)[:target_vocab_size] # decending order, int
    new_vocab = [_vocab[i] for i in vocab_idx_wanted]
    new_vocab_dict = dict(zip(new_vocab, range(target_vocab_size)))
    with open("topic_model_vocab.txt", 'w') as w_f:
        w_f.write('\n'.join(new_vocab))
    return new_vocab, new_vocab_dict

def read_topic_data(path, reduced_vocab):
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    myFile= open(path, "rU" )
    d = []
    for i, aRow in enumerate(myFile):
        x = aRow.split()[1]
        d.append(x)
    myFile.close()
    counts = count_vect.fit_transform(d).toarray()
    doc_word_sum = np.sum(counts, axis=1)
    valid_idx = doc_word_sum>5
    x = counts[valid_idx]
    return x, valid_idx

def read_topical_atten_data(path, vocab_dict, reduced_vocab):
    myFile= open(path, "rU")
    labels = []
    docIDs = []
    docs = []
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    for i, aRow in enumerate(myFile):
        line = aRow.strip().split()
        ids = []
        for w in line[1].split(','):
            if w in vocab_dict:
                ids.append(vocab_dict[w])
        if len(ids)>5:
            labels.append(class_dict[line[0]])
            docIDs.append(ids)
            docs.append(line[1])
    myFile.close()
    num_docs = len(labels)
    print(num_docs, "docs in total")
    y = np.zeros((num_docs, num_classes), dtype=np.int32)
    x = np.zeros((num_docs, MAX_NUM_WORD), dtype=np.int32)
    for i in range(num_docs):
        y[i][labels[i]] = 1
        if len(docIDs[i])>MAX_NUM_WORD:
            x[i, :] = docIDs[i][:MAX_NUM_WORD]
        else:
            x[i, :len(docIDs[i])] = docIDs[i]
    counts = count_vect.fit_transform(docs).toarray()
    return x, y, counts, num_docs    