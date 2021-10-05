from sklearn.preprocessing import Normalizer
import numpy as np

class TopicCollection:
    def __init__( self, top_terms = 0, threshold = 1e-6 ):
        # settings
        self.top_terms = top_terms
        self.threshold = threshold
        # state
        self.topic_ids = []
        self.all_weights = []
        self.all_terms = set()

    def add_topic_model( self, H, terms, window_topic_labels ):
        '''
        Add topics from a window topic model to the collection.
        '''
        # k = get number of topics rows
        k = H.shape[0]
        for topic_index in range(k): # for every topic index
            topic_weights = {}
            
            # use top terms only (sparse topic representation)?
            
            if self.top_terms > 0: # if only taking some number of top terms
                top_indices = np.argsort( H[topic_index,:] )[::-1] # arrange term values in descending order
                for term_index in top_indices[0:self.top_terms]: # take the top terms
                    topic_weights[terms[term_index]] = H[topic_index,term_index] # add key = term, value = weight
                    self.all_terms.add( terms[term_index] ) # add to total term vocab the term
                    
            # use dense window topic vectors
            else: # if not using a top term
                # calculate the sum of all term weights (should be 1?)
                total_weight = 0.0
                for term_index in range(len(terms)):
                    total_weight += H[topic_index,term_index]
                for term_index in range(len(terms)): # get the proportion of the topic distribution this word accounts for
                    w = H[topic_index,term_index] / total_weight
                    if w >= self.threshold: # if above the threshold value
                        topic_weights[terms[term_index]] = H[topic_index,term_index] # add term to top_weights dict
                        self.all_terms.add( terms[term_index] ) # add to entire vocab
            self.all_weights.append( topic_weights ) # add the topic weight dictionary
            self.topic_ids.append( window_topic_labels[topic_index] ) # add the topic names

    def create_matrix( self ):
        '''
        Create the topic-term matrix from all window topics that have been added so far.
        '''
        # map terms to column indices
        all_terms = list(self.all_terms)
        M = np.zeros( (len(self.all_weights), len(all_terms)) )
        term_col_map = {}
        for term in all_terms:
            term_col_map[term] = len(term_col_map)
        # populate the matrix in row-order
        row = 0
        for topic_weights in self.all_weights:
            for term in topic_weights.keys():
                M[row,term_col_map[term]] = topic_weights[term]
            row +=1
        # normalize the matrix rows to L2 unit length
        normalizer = Normalizer(norm='l2', copy=True)
        normalizer.fit(M)
        M = normalizer.transform(M)
        return (M,all_terms)