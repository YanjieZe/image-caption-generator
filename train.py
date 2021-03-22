from model import ShowTellNet, one_hot_encoder, LossNet
import torch
import torch.nn as nn
import pickle as pkl
from dataloader import VisDialDataset, visdial_collate_fn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch.optim import Adam,SGD
from tensorboardX import SummaryWriter
import logging


def train(
            batch_size=8, 
            epoch=10,
            learing_rate=1e-4,
            tensorboard_path='./tensorboard_result',
            log_path='./log/train5.log',
            model_param_savepath='./model_param',
            model_param_loadpath='model_param/epoch19.pth'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, filename=log_path)

    writer = SummaryWriter(tensorboard_path)
    train_loader = DataLoader(VisDialDataset(None, 'train'), collate_fn=visdial_collate_fn,
                              batch_size=batch_size, shuffle=True, num_workers=4)

    val_length = len(VisDialDataset(None, 'val'))

    model = ShowTellNet(hidden_sz=512)
    if model_param_loadpath:
        model.load_state_dict(torch.load(model_param_loadpath))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learing_rate,betas=(0.9,0.99),eps=1e-8)
    # optimizer = SGD(model.parameters(), lr=learing_rate)
    loss_fn = LossNet().to(device)
    for eps in range(epoch):
        
        model.train()
        logging.info("train epoch: %d"%eps)
        for cnt, batched in enumerate(train_loader):
            cap = batched['captions'].to(device)
            img = batched['features'].to(device)
            # one_hot_cap = one_hot_encoder(cap).to(device)
            seq_prob, (h_t, c_t)=model(img=img, x=cap)

            loss = loss_fn(cap, seq_prob)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt%50==0:
                logging.info("count: %d, loss:%f: "%(cnt,loss.item()))
                writer.add_scalar('Train_Loss', loss, cnt)

        # save model of an epoch        
        torch.save(model.state_dict(), model_param_savepath+"/epoch%d.pth"%(eps+18))
        
            

if __name__=="__main__":
    train()