from scipy.sparse import csr_matrix
from MyNLPToolBox import TextPreprocessor as TP
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
import sys

class TFIDFVectorizer:
    def __init__(self,mode='log'):
        self.dict = []
        self.dict_uniq = []
        self.tf_mat = None # Store the tf mat for later use to calculate idf
        self.mode = mode
        
    def change_mode(self,mode):
	"""
	Change to different modes of calculate tfidf:
	'natural'
	'log'
	'boolean'
	'augmented'
	"""
        self.mode = mode
        return self

    def fit(self,corpus):
        for words in corpus:
            self.dict += words.split()
        self.dict_uniq = list(np.unique(self.dict))
        return self

        
    def transform(self,corpus):
        
        # Initialize variable
        tf = None
        idf = None
        
        # Choose between tf modes
        if self.mode == 'natural':
            tf = self.tf(corpus)
        elif self.mode == 'log':
            tf = self.logtf(corpus)
        elif self.mode == 'boolean':
            tf = self.booleantf(corpus)
        elif self.mode == 'augmented':
            tf = self.augtf(corpus)
            
        # Calculate idf
        idf = self.idf(corpus)
        
        # Loading bar
        print('Performing Tfidf..',end='')
        
        # Calculate tfidf = tf*idf
        tfidf = tf.multiply(idf)
        
        print(' Done!')
        
        return tfidf
        
    

    def tf(self,corpus): 
        
        # (Natural) tf is define number of occurence of a word in a doc 
        data = [] 
        row = []
        col = []
        for idx,sentence in enumerate(corpus):
            
            # Loading bar
            if idx%500==0:
                sys.stdout.write('\r'+'Performing Tf..'+str(round(idx/len(corpus)*100))+'%')

            # Tokenize the sentence
            words_list = sentence.split()
            
            for word in set(words_list):
                #If the word is not in our dictionary then skip
                if word not in self.dict_uniq: continue
                # Frequency of that word in the same sentence
                data.append(words_list.count(word)) 
                # Row of the word, equivalent to index of that sentence (holding that word) in the corpus
                row.append(idx)
                # Col of the word, equivalent to the unique index of the word in the dictionary
                col.append(self.dict_uniq.index(word))
                
        print(' Done!')
        
        # Create sparse matrix from collective variables
        tf_mat = csr_matrix((data, (row, col)), shape=(len(corpus), len(self.dict_uniq)))
        
        # Save the tf_mat for later use to calculate idf
        self.tf_mat = tf_mat
       
        return tf_mat
   

    def logtf(self,corpus):
        # Logarithmic tf is defined as 1 + log2(tf)
        
        # Retrieve term frequency from "tf" method
        log_matrix = self.tf(corpus)
        
        # Loading bar
        print('Performing LogTf..',end='')
        
        # Perform log-tf
        log_matrix.data = 1+np.log2(log_matrix.data)
        
        print(' Done!')
        
        return log_matrix
    
    
    def booleantf(self,corpus):
        # Boolean tf is defined as 1 when tf>1 and 0 otherwise
        
        # Retrieve term frequency from "tf" method
        boolean_matrix = self.tf(corpus)
        
        # Loading bar
        print('Performing BooleanTf..',end='')
        
        # Perform boolean-tf
        boolean_matrix.data[boolean_matrix.data > 0] = 1
        boolean_matrix.data[boolean_matrix.data = 0] = 0
        
        print(' Done!')
        
        return boolean_matrix
    
    def augtf(self,corpus):
        # Augmented tf is defined as 0.5 + 0.5x tf(word) /max(tf(word) in corpus)
        
        # Retrieve term frequency from "tf" method
        aug_matrix = self.tf(corpus).astype(np.float)
        for i in range(len(corpus)-1):
            # Loading bar
            if i%500==0:
                sys.stdout.write('\r'+'Performing AugTf..'+str(round(i/len(corpus)*100))+'%')
            
            # Retrieve i-th row of matrix
            row = aug_matrix.data[aug_matrix.indptr[i]:aug_matrix.indptr[i+1]]
            
            # Get the max value of that row
            maxtf = row.max()
            
            # Compute Augmented TF
            row = 0.5 + 0.5*row/maxtf
            
            # Re-assign the matrix
            aug_matrix.data[aug_matrix.indptr[i]:aug_matrix.indptr[i+1]] = row
            
        print(' Done!')
        
        return aug_matrix
        
    def idf(self,corpus):
        # (Logarithmic) tf is defined as log2(num_docs/num_docs_have_word)
        # Loading bar
        print('Performing Idf..',end='')
        
        # Initialize variables
        num_docs = len(corpus) # Number of docs
        num_features = len(self.dict_uniq) # Number of features
        
        # Retrieve term frequency from save tf_mat and convert to csc for column-wise manipulating
        idfmat = self.tf_mat.tocsc().astype(np.float)
        
        for i in range(num_features):
            
            # array of i-th word weights
            weights = idfmat.data[idfmat.indptr[i]:idfmat.indptr[i+1]]
            
            # Number of docs that has i-th word
            num_docs_have_word =  np.count_nonzero(weights)
            try:
                # Perform idf
                weights = np.log2(num_docs/num_docs_have_word)

                # Re-assign to our matrix
                idfmat.data[idfmat.indptr[i]:idfmat.indptr[i+1]] = weights
            except ZeroDivisionError:
                pass
            
        print(' Done!')
        
        # Switch back to csc mat
        idfmat.tocsr()
        
        return idfmat
        
