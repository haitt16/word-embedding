import argparse
import numpy as np
from doc_tool import DocTool
import logging

class CoOccurenceMatrix(DocTool):
    def __init__(self, document_path: str, 
                        window_size: int = 1, 
                        direction: str = 'around'):
        self.doc_path = document_path
        self.window_size = window_size
        self.direction = direction
        self.doc_data = self.read(document_path)
        self.amount_data = len(self.doc_data)
        self.word2index = {w: i for w, i in enumerate(self.getInstances(self.doc_data))}
        self.co_occurence_matrix = np.zeros((len(self.all_term),
                                                len(self.all_term))) 
        self.createMatrix()

    def updateMatrix(self, target: str, contexts: list):
        fi = self.word2index[target]
        for context in contexts:
            fo = self.word2index[context]
            self.co_occurence_matrix[fi][fo] += 1
        

    def getContext(self, sample: str):
        tracking_list = sample.split()
        sliding_list = [0]*self.window_size + tracking_list + \
                                    [0]*self.window_size
        for idx, target in enumerate(tracking_list, self.window_size):
            if self.direction == 'around':
                contexts: list = sliding_list[idx-self.window_size : idx] \
                            + sliding_list[idx+1 : idx+self.window_size] 
            elif self.direction == 'front':
                contexts: list = sliding_list[idx-self.window_size : idx]
            elif self.direction == 'back':
                contexts: list = sliding_list[idx+1 : idx+self.window_size]
            else:
                logging.warning('Your input direction is not valid. \
                            You should try 1 of 3 options [\'around\', \'front\', \'back\'')
            while 0 in contexts:
                contexts.remove(0)
            self.updateMatrix(target, contexts)
    
    def createMatrix(self):
        for sample in self.doc_data:
            self.getContext(sample)

def main(args):
    word_embedded = CoOccurenceMatrix(args.path).co_occurence_matrix
    print(word_embedded[args.term])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='input data path', dest='path',
                            default='./tf_idf_data.txt', type=str)
    parser.add_argument('--term', help='word that you wanna see its vector', dest='term',
                            default="handsome", type=str)
    args = parser.parse_args()
    main(args)        
