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
import logging


def test(
            batch_size=8, 
            epoch=5,
            result_path='./result',
            model_param_filepath='./model_param/epoch4.pth'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, filename='./log/test.log')

    writer = SummaryWriter(result_path)
    
    test_loader = DataLoader(VisDialDataset(None, 'val'), collate_fn=visdial_collate_fn,
                              batch_size=batch_size, shuffle=True, num_workers=4)
    val_length = len(VisDialDataset(None, 'val'))

    model = ShowTellNet().to(device)
    
    model.load_state_dict(torch.load(model_param_filepath))
    for eps in range(epoch):  

        model.eval()
        logging.info("test epoch: %d"%eps)
        for cnt, batched in enumerate(test_loader):
            cap = batched['captions'].to(device)
            img = batched['features'].to(device)
            one_hot_cap = one_hot_encoder(cap).to(device)
            seq_prob, (_, _)=model(img, one_hot_cap)
            cap_pred = torch.argmax(seq_prob,2)

            # calculate accuracy
            total_count = cap.shape[0]*cap.shape[1]
            acc_count = (cap_pred==cap).sum().item()/total_count
            if cnt%50==0:
                writer.add_scalar('Test_Accuracy', acc_count, cnt)    
                logging.info("count: %d, acc:%f: "%(cnt, acc_count))
  
            


if __name__=="__main__":
    test()