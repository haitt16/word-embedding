import argparse
import numpy as np
from doc_tool import DocTool, unique
from tf_idf import TF_IDF

class TF_IDF_Vector(TF_IDF):
    def __init__(self, document_path: str):
        self.doc_path = document_path
        self.doc_data = self.read(document_path)
        self.amount_data = len(self.doc_data)
        self.all_term = self.getInstances(self.doc_data)
        self.word_vec = self.init_word_vec(self.amount_data, self.all_term)
        self.merge_to_vector()
    
    def merge_to_vector(self):
        for idx, sample in enumerate(self.doc_data):
            tf_idf_dict = self.tf_idf(sample)
            for term in tf_idf_dict.keys():
                self.word_vec[term][idx] = tf_idf_dict[term]

def main(args):
    word_embedded = TF_IDF_Vector(args.path).word_vec
    print(word_embedded[args.term])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='input data path', dest='path',
                            default='./tf_idf_data.txt', type=str)
    parser.add_argument('--term', help='word that you wanna see its vector', dest='term',
                            default="handsome", type=str)
    args = parser.parse_args()
    main(args)