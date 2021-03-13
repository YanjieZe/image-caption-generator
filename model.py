import torch
import torch.nn as nn
import pickle as pkl
from dataloader import VisDialDataset, visdial_collate_fn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

def index2word(index:torch.tensor):
    """
    index: batch_size * 24
    """
    with open('data/cache/ind2word_32525.pkl', 'rb') as f:
        ind2word = pkl.load(f) # total 32529 word

    index_np = index.numpy()
    sentences = []
    for batch in range(index_np.shape[0]):
        sentence = []
        for i in range(index_np.shape[1]):
            sentence.append(ind2word[index_np[batch,i]])
        sentences.append(sentence)
    return sentences


def word2index(sentence):
    with open('data/cache/word2ind_32525.pkl', 'rb') as f:
        word2ind = pkl.load(f)
    pass


def one_hot_encoder(index:torch.tensor):
    """
    func: transfer index(caption) into one hot
    return: batch_size * sentence_lenght * 32529
    """
    index_one_hot = torch.zeros(index.shape[0], index.shape[1],32529)
    for batch in range(index.shape[0]):
        for i in range(index.shape[1]):
            ind = index[batch, i]
            index_one_hot[batch, i, ind] = 1
    return index_one_hot


class CNN(nn.Module):
    def __init__(self, input_channels=4096):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=2048,kernel_size=3)
        self.bn1 = nn.BatchNorm1d(2048)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(1024)
        

    def forward(self):
        pass



class LSTM(nn.Module):
    
    def __init__(self, input_sz:int, hidden_sz:int):
        super(LSTM, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        
        #i_t
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        
        #f_t
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        
        #c_t
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        
        #o_t
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    

    def forward(self,
                x,
                init_states=None):
        
        """
        x.shape represents (batch_size, sequence_size, input_size)
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states
            
        for t in range(seq_sz):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
        
        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class ShowTellNet(nn.Module):
    def __init__(self, input_sz=32529, hidden_sz=256, output_sz=32529,image_feature_sz=4096):
        super(ShowTellNet, self).__init__()
        self.hidden_size = hidden_sz
        self.LSTM = LSTM(input_sz=hidden_sz, hidden_sz=hidden_sz)
        self.linear_in = nn.Linear(input_sz, hidden_sz)
        self.linear_out = nn.Linear(hidden_sz, output_sz)
        self.linear_img = nn.Linear(image_feature_sz, 14*hidden_sz)
        self.softmax = nn.Softmax(2)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, image_feature, one_hot_caption):

        x = self.linear_in(one_hot_caption) # batch * sentence_length * 4096

        batch_size = image_feature.shape[0]
        feature = self.linear_img(image_feature).reshape(batch_size, 14, self.hidden_size)

        # using image feature
        _, (h_0,c_0) = self.LSTM(feature)

        # using caption
        hid_seq, (h_t, c_t) = self.LSTM(x, init_states=(h_0,c_0))
        
        hid_seq = self.linear_out(hid_seq)
        
        seq_prob = self.softmax(hid_seq)

        return seq_prob,(h_t, c_t)
        
class LossNet(nn.Module):
    
    def __init__(self):
        super(LossNet, self).__init__()

    def forward(self, cap_index, pred):
        loss_sum=0
        for batch in range(cap_index.shape[0]):
            for i in range(cap_index.shape[1]):
                ind = cap_index[batch, i]
                loss_sum = loss_sum - torch.log(pred[batch, i, ind])
        return loss_sum


if __name__=="__main__":
    train_loader = DataLoader(VisDialDataset(None, 'train'), collate_fn=visdial_collate_fn,
                              batch_size=4, shuffle=True, num_workers=4)
    net = ShowTellNet()
    loss_fn = LossNet()
    for cnt, batched in enumerate(train_loader):
        cap = batched['captions']
        img = batched['features']
        one_hot_cap = one_hot_encoder(cap)
        seq_prob, (h_t, c_t)=net(img, one_hot_cap)
        loss = loss_fn(cap, seq_prob)
        print(loss)
        break
