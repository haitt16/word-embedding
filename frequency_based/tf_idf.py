import argparse
import logging
import os
import math
from doc_tool import DocTool


class TF_IDF(DocTool):
    def __init__(self, document_path: str):
        self.path = document_path
        self.doc_data = self.read(document_path)
        self.data_count = len(self.doc_data)
    
    def tf(self, doc: str):
        term_dict = self.term_counting(doc)
        doc_length = len(doc.split())
        for term in term_dict.keys():
            term_dict[term] /= doc_length
        
        return term_dict
    
    def idf(self, doc: str):
        term_dict = self.term_counting(doc)
        terms = term_dict.keys()
        idf_value = {}

        for term in terms:
            idf_value[term] = 0

        for term in terms:
            for sample in self.doc_data:
                if term in sample:
                    idf_value[term] += 1
        
        for term in idf_value.keys():
            idf_value[term] = math.log2(self.data_count / idf_value[term])
        
        return idf_value

    def tf_idf(self, doc: str):
        tf = self.tf(doc)
        idf = self.idf(doc)
        
        tf_idf = {}
        terms = tf.keys()

        for term in terms:
            tf_idf[term] = tf[term] * idf[term]
        
        return tf_idf

def main(args):
    calculator = TF_IDF(args.path)
    doc = calculator.doc_data[args.idx]
    print(f'tf-idf of "{doc}" is {calculator.tf_idf(doc)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='input data path', dest='path',
                            default='./tf_idf_data.txt', type=str)
    parser.add_argument('--index', help='index of a sample in dataset you wanna cal', dest='idx',
                            default=3, type=int)
    args = parser.parse_args()
    main(args)
    






        


