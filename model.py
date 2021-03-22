import torch
import torch.nn as nn
import pickle as pkl
from dataloader import VisDialDataset, visdial_collate_fn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


def index2word(index:torch.tensor):
    """
    index: batch_size * sentence_length
    """
    with open('./data/cache/ind2word_32525.pkl', 'rb') as f:
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
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=2048,kernel_size=1)
        self.bn1 = nn.BatchNorm1d(2048)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=1024,kernel_size=1)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            weight.data.uniform_(0, 1)

    def forward(self, x):
        batch_szie = x.shape[0]
        x = x.reshape(batch_szie, 4096,1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x.reshape(batch_szie, 512)
        return x

    
class ShowTellNet(nn.Module):
    
    def __init__(self, origin_size=32529, input_sz=512, hidden_sz=512, img_feature_sz=4096):
        super(ShowTellNet, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        
        self.embedding = nn.Embedding(origin_size, hidden_sz)
        #self.bn1 = nn.BatchNorm1d(input_sz)
        self.linear_out = nn.Linear(hidden_sz, origin_size)
        self.cnn = CNN()
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
        
        self.softmax = nn.Softmax(2)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    

    def forward(self,
                img,
                x,
                init_states=None):
        
        """
        x.shape represents (batch_size, sequence_size, input_size)
        """
        img_feature = self.cnn(img)
        #x = self.linear_in(x)
        x = self.embedding(x)
        #print(x.shape)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        x_t = img_feature.reshape(bs, self.hidden_size)
        i_t = torch.sigmoid(img_feature + h_t @ self.V_i + self.b_i)
        f_t = torch.sigmoid(img_feature + h_t @ self.V_f + self.b_f)
        g_t = torch.tanh(img_feature + h_t @ self.V_c + self.b_c)
        o_t = torch.sigmoid(img_feature + h_t @ self.V_o + self.b_o)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)   
        # hidden_seq.append(h_t.unsqueeze(0)) 

        for t in range(seq_sz):
            x_t = x[:, t, :] # 4*512
            # print(x_t.shape)
            #x_t = self.bn1(x_t)
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)# 4* 512

            hidden_seq.append(h_t.unsqueeze(0))
        
        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        hidden_seq = self.linear_out(hidden_seq)

        seq_pred = self.softmax(hidden_seq)
        return seq_pred, (h_t, c_t)


    def predict(self, img:torch.tensor, init_states=None):

        img_feature = self.cnn(img)
        #x = self.linear_in(x)
     
        #print(x.shape)
        bs, _ = img_feature.size()
        seq_sz = 24
        
        x = torch.ones(bs)*32526
        x_t = self.embedding(x.long())# 4*512
        
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        x_t = img_feature.reshape(bs, self.hidden_size) # 4 * 512
        
        i_t = torch.sigmoid(img_feature + h_t @ self.V_i + self.b_i)
        f_t = torch.sigmoid(img_feature + h_t @ self.V_f + self.b_f)
        g_t = torch.tanh(img_feature + h_t @ self.V_c + self.b_c)
        o_t = torch.sigmoid(img_feature + h_t @ self.V_o + self.b_o)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)   
        # hidden_seq.append(h_t.unsqueeze(0)) 

        for t in range(seq_sz):
           
            #x_t = self.bn1(x_t)
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t) # 4 * 512
            
            word_pred = self.linear_out(h_t) # 4 * 32529
            word_pred = torch.softmax(word_pred, dim=1)
            word_pred = torch.argmax(word_pred, dim=1).long()
            
            # next input
            x_t = self.embedding(word_pred)

            hidden_seq.append(word_pred.unsqueeze(0))
        
        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq
   
        
        
class LossNet(nn.Module):
    
    def __init__(self):
        super(LossNet, self).__init__()

    def forward(self, cap_index, pred):
        loss_sum=0
        
        for batch in range(cap_index.shape[0]):
            num = cap_index.shape[1]
            for i in range(cap_index.shape[1]):
                if(i+1==cap_index.shape[1]):
                    break
                ind = cap_index[batch, i+1]
                loss_sum = loss_sum - np.square(1-i/num)*torch.log(pred[batch, i, ind])
        return loss_sum


if __name__=="__main__":
    train_loader = DataLoader(VisDialDataset(None, 'train'), collate_fn=visdial_collate_fn,
                              batch_size=4, shuffle=True, num_workers=4)
    net = ShowTellNet()
    loss_fn = LossNet()
    net.load_state_dict(torch.load("model_param/epoch18.pth"))
    # train
    for cnt, batched in enumerate(train_loader):
        cap = batched['captions']
        img = batched['features']
        #one_hot_cap = one_hot_encoder(cap)
        seq_prob, (h_t, c_t)=net(img, cap)
        loss = loss_fn(cap, seq_prob)
        print(loss)
        break

    # predict
    """
    for cnt, batched in enumerate(train_loader):
        cap = batched['captions']
        img = batched['features']
        #one_hot_cap = one_hot_encoder(cap)
        real_sentence=net.predict(img)
        break
    """