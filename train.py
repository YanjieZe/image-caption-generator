from model import ShowTellNet, one_hot_encoder, LossNet
import torch
import torch.nn as nn
import pickle as pkl
from dataloader import VisDialDataset, visdial_collate_fn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch.optim import Adam
from tensorboardX import SummaryWriter


def train(
            batch_size=8, 
            epoch=5,
            learing_rate=1e-3,
            result_path='./result'):

    writer = SummaryWriter(result_path)
    train_loader = DataLoader(VisDialDataset(None, 'train'), collate_fn=visdial_collate_fn,
                              batch_size=batch_size, shuffle=True, num_workers=4)
    model = ShowTellNet()
    optimizer = Adam(model.parameters(), lr=learing_rate)
    loss_fn = LossNet()
    for eps in range(epoch):
        for cnt, batched in enumerate(train_loader):
            cap = batched['captions']
            img = batched['features']
            one_hot_cap = one_hot_encoder(cap)
            seq_prob, (_, _)=model(img, one_hot_cap)
            loss = loss_fn(cap, seq_prob)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt%20==0:
                print("epoch: %d, count: %d, loss:%f: "%(eps, cnt,loss.item()))
                writer.add_scalar('loss', loss, cnt)
            



if __name__=="__main__":
    train()