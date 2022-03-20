import numpy as np
import torch
import torch.nn.functional as F
from doc_tool import DocTool
import logging
import random


class Dataset(DocTool):
    def __init__(self, document_path: str, 
                        window_size: int = 1, 
                        direction: str = 'around'):
        self.doc_path = document_path
        self.window_size = window_size
        self.direction = direction
        self.doc_data = self.read(document_path)
        self.amount_data = len(self.doc_data)
        self.word2index = {w: i for i, w in enumerate(self.getInstances(self.doc_data))}
        self.dataloader = []
        self.getDataloader()
    
    def __len__(self):
        return len(self.dataloader)

    def one_hot_encoder(self, value: int):
        n = len(self.word2index)
        one_hot_vec = torch.zeros(n)
        one_hot_vec[value] = 1
        return one_hot_vec
 
    def getPairs(self, target: str, contexts: str):
        for context in contexts:
            tgt = self.one_hot_encoder(self.word2index[target])
            ctxt = self.one_hot_encoder(self.word2index[context])
            self.dataloader.append((tgt, ctxt))

    def getContexts(self, sample: str):
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
            self.getPairs(target, contexts)
    
    def getDataloader(self):
        for sample in self.doc_data:
            self.getContexts(sample)
    
        
        
