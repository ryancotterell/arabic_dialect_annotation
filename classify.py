import csv
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.svm import LinearSVC,SVC
from sklearn.cross_validation import KFold

def main():
    data = []
    num = int(sys.argv[1])
    folds = 10
    examples = [ [] for x in range(0,10)]
    results = []
    for i,f in enumerate(sys.argv[2:]):
        data.append(csv.DictReader(open(f,'rb'),delimiter='\t'))
    for f in data:       
        for i,datum in enumerate(f):
            examples[i % folds].append(datum)

    for held_out in range(0,folds):
        labels = []
        corpus = []
        labels_test = []
        corpus_test = []
        vectorized = []
        vectorized_test = []

        for i,fold in enumerate(examples):
            for line in fold:
                if i == held_out:
                    labels_test.append(line['label'].rstrip("\n"))
                    corpus_test.append(line['text'].rstrip("\n"))
                else:
                    labels.append(line['label'].rstrip("\n"))
                    corpus.append(line['text'].rstrip("\n"))
                
        vectorizer = CountVectorizer(ngram_range=(1,1),min_df=1)
        X = vectorizer.fit_transform(corpus)
        for c in corpus:        
            tmp = vectorizer.transform([c]).toarray()
            vectorized.append(tmp[0])
        for c in corpus_test:        
            tmp = vectorizer.transform([c]).toarray()
            vectorized_test.append(tmp[0])

        classifier = MultinomialNB()
        #classifier = SVC(C=1.0,kernel='rbf')
        classifier.fit(vectorized,labels)
        result = accuracy(labels_test,vectorized_test,classifier)

        print "Fold %d %f" % (held_out,result)
        results.append(result)

    print sum(results) / len(results)

def accuracy(labels,vectorized,classifier):
    count = 0
    total = 0
    for l,v in zip(labels,vectorized):
        if l ==  classifier.predict(v)[0]:
            count += 1
        total += 1
    return float(count)/total

if __name__ == "__main__":
    main()
