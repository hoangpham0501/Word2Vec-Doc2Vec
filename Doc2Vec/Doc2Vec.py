# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# Read data in *.txt format and yield to object Labeled Sentence
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources       
        flipped = {}        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

# Load Data
sources = {
    'data/test-neg.txt':'TEST_NEG',
    'data/test-pos.txt':'TEST_POS', 
    'data/train-neg.txt':'TRAIN_NEG', 
    'data/train-pos.txt':'TRAIN_POS', 
    'data/train-unsup.txt':'TRAIN_UNS'
}

sentences = LabeledLineSentence(sources)

# Model
# min_count: remove the words from dictionary which have the number of occurence < min_count
# window: max distance of current word and predicted word
# size: dimension of the embedded vector (100-400)
# workers: number of worker thread (= number of cores of machine)
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=3)
model.build_vocab(sentences.to_array())

#Training
# each "epochs" = the number of train
model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=10)

# Saving model
model.save('./imdb.d2v')



