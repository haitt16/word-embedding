from numpy import require
import torch
import torch.nn.functional as F
from dataloader import Dataset
from torch.autograd import Variable

class SkipGramModel:
    def __init__(self, learning_rate: float = 0.001, 
                        embedding_dim: int = 5,
                        vocab_size: int = None):
        self.lr = learning_rate
        self.emb_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_hidden_Weight = Variable(torch.rand(embedding_dim, vocab_size).float(), 
                                    requires_grad=True)
        self.output_hidden_Weight = Variable(torch.rand(vocab_size, embedding_dim).float(),
                                    requires_grad=True)

    def forward(self, x):
        z = torch.matmul(self.input_hidden_Weight, x)
        y_hat = torch.matmul(self.output_hidden_Weight, z)
        return y_hat
    
    def updateWeight(self):
        self.input_hidden_Weight.data \
                        -= self.lr * self.input_hidden_Weight.grad.data
        self.output_hidden_Weight.data \
                        -= self.lr * self.output_hidden_Weight.grad.data
        self.input_hidden_Weight.grad.data.zero_()
        self.output_hidden_Weight.grad.data.zero_()

def train(args):
    dataset = Dataset(args.path, 
                    args.window_size).dataloader

    model = SkipGramModel(args.lr,
                        args.emb_dim, 
                        len(dataset))
    for epoch in range(args.num_epochs):
        loss_val = 0
        for target, context in dataset:
            x = Variable(target).float()
            y = Variable(context).float()

            y_hat = model.forward(x)
            y_prob = F.log_softmax(y_hat, dim=0)
            
            loss = F.nll_loss(y_prob.view(1,-1), y)
            loss_val += loss.item()
            loss.backward()

            model.updateWeight()
        with open("log_loss.txt", "a") as f:
            f.write(f'{epoch}\t\t\t{loss_val/len(dataset)}')             

    return model.input_hidden_Weight

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='input data path', dest='path',
                            default='./tf_idf_data.txt', type=str)
    parser.add_argument("--window_size", help='Context window size', dest='window_size',
                            default=1, type=int)
    parser.add_argument('--lr', help='learning rate', dest='lr',
                            default=0.0001, type=float)
    parser.add_argument('--emb_dim', help='embedding dimension', dest='emb_dim',
                            default=5, type=int)
    parser.add_argument('--num_epochs', help='number of training epochs', dest='num_epochs',
                            default=100, type=int)
    args = parser.args()
    word_embedding = train(args)