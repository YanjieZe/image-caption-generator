from model import ShowTellNet
import torch
from dataloader import VisDialDataset, visdial_collate_fn
from torch.utils.data import Dataset, DataLoader
import pickle as pkl

def index2word(index:torch.tensor):
    """
    index: batch_size * 24
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



if __name__=='__main__':
    model = ShowTellNet()
    model.load_state_dict(torch.load("./model_param/epoch27.pth"))

    test_loader = DataLoader(VisDialDataset(None, 'val'), collate_fn=visdial_collate_fn,
                              batch_size=2, shuffle=True, num_workers=4)

    for cnt, batched in enumerate(test_loader):
        img = batched['features']
        cap = batched['captions']
        cap_pred = model.predict(img)
        
        # print("pred: ", cap_pred)
        # print("origin: ", cap)
        sentence_origin = index2word(cap)
        sentence_pred = index2word(cap_pred)
        print("pred: ", sentence_pred)
        print("origin: ", sentence_origin)

        break

