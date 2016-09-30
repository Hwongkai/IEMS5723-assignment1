import nltk

#Initialize the training data (restaurants and labels)
train = [0, 0, 0]
train[0] = nltk.word_tokenize("Chinese food rice rice noodles")
train[1] = nltk.word_tokenize("Chinese Chinese rice food")
train[2] = nltk.word_tokenize("Italian pizza rice restaurant")
label = ['C','C','I']

#Self-define function, which returns the frequency distribution of restaurant words in documents
def document_features(documents): 
    fdist = nltk.FreqDist(documents)
    return fdist

#documents = training data + label, print them to see the contents
documents = []
for a in range(0,len(train)) :
	documents.append((train[a],label[a]))
print documents[0]
print documents[1]
print documents[2]

#feature = frequency distribution of restaurant words in documents + label
doc_feat = []
for a in range(0,len(train)) :
	doc_feat.append((document_features(documents[a][0]),documents[a][1]))
		
#To train the Naive Bayes classifier using training data, print the training data accuracy
classifier = nltk.NaiveBayesClassifier.train(doc_feat)
print(nltk.classify.accuracy(classifier, doc_feat))

#To classify a new testing data using the newly-trained classifier (get a label)
test = nltk.word_tokenize("Chinese Chinese food noodles restaurant")
print classifier.classify(document_features(test))