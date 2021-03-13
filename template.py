import pickle as pkl
import random
import torch
import torch.nn as nn
import torch.optim as optim

from options import params
from dataloader import VisDialDataset, visdial_collate_fn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

# Generate sentence from tokens
with open('data/cache/word2ind_32525.pkl', 'rb') as f:
    word2ind = pkl.load(f)
with open('data/cache/ind2word_32525.pkl', 'rb') as f:
    ind2word = pkl.load(f)

# Seed rng for reproducibility
random.seed(params.seed)
torch.manual_seed(params.seed)
if params.cuda:
    torch.cuda.manual_seed_all(params.seed)

# Setup dataloader
splits = ['train', 'val', 'test']
train_loader = DataLoader(VisDialDataset(params, 'train'), collate_fn=visdial_collate_fn,
                          batch_size=params.batch_size, shuffle=True, num_workers=params.n_works)
val_loader = DataLoader(VisDialDataset(params, 'val'), collate_fn=visdial_collate_fn,
                        batch_size=params.batch_size, shuffle=True, num_workers=params.n_works)


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------


def batch_data(entry):
    features = entry['features']
    captions = entry['captions'].long()
    questions = entry['questions'].long()
    answers = entry['answers'].long()
    caption_msks = entry['caption_msks']
    question_msks = entry['question_msks']
    answer_msks = entry['answer_msks']
    scene_id = entry['scene_id']
    image_id = entry['image_id']
    if params.cuda:
        features = features.cuda(non_blocking=True)
        captions = captions.cuda(non_blocking=True)
        # questions = questions.cuda(non_blocking=True)
        # answers = answers.cuda(non_blocking=True)
        # caption_msks = caption_msks.cuda(non_blocking=True)
        # question_msks = question_msks.cuda(non_blocking=True)
        # answer_msks = answer_msks.cuda(non_blocking=True)

    return scene_id, image_id, features, captions, questions, answers, caption_msks, question_msks, answer_msks


for epoch in range(params.epochs):
    for idx, batch in enumerate(train_loader):
        # Moving current batch to GPU, if available
        scene_id, image_id, image, captions, _, _, caption_msks, _, _ = batch_data(batch)
        caption_lens = caption_msks.sum(dim=-1).cuda(non_blocking=True)
