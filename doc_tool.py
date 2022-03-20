import numpy as np
from natsort import natsorted
class DocTool:
    def read(self, document_path: str):
        # This function process to read doc in .txt file. Each sample is saved in a single line
        with open(document_path, 'r') as f:
            doc_data = f.readlines()
        
        doc_data = [i[:-1].lower() for i in doc_data if '\n' in i]
        
        return doc_data

    def term_counting(self, doc: str):
        doc = doc.split()
        term_dict = {}
        for term in doc:
            if term in term_dict.keys():
                term_dict[term] += 1
            else:
                term_dict[term] = 1
        return term_dict 
    
    def getInstances(self, doc_data):
        all_data = ' '.join(doc_data).split()
        all_data = unique(all_data)
        return natsorted(all_data)

    def init_word_vec(self, dimension: int, all_term: list):
        word_vec = {}
        vector = np.zeros(dimension)
        for term in all_term:
            word_vec[term] = vector
        return word_vec

def unique(list: list):
    unique_list = []
    unq = set(list)
    for i in unq:
        unique_list.append(i)
    
    return unique_list
