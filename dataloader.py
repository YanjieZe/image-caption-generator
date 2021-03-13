import os

# os.environ['NLTK_DATA'] = '/mnt/jdwu_hdd/nltk_data'

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VisDialDataset(Dataset):
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self.subset = subset
        with open('/mnt/VisDial_hdd/VisualDialog_' + subset + '.pkl', 'rb') as f:
            self.db = pickle.load(f)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, scene_id):
        scene = self.db[scene_id]

        # global features
        image_id = scene['image_id']
        with open('/mnt/VisDial_hdd/Vgg16_' + self.subset + '/' + str(image_id) + '.pkl', 'rb') as f:
            feature = pickle.load(f)

        caption = scene['caption']
        quests = scene['quests']
        answers = scene['answers']

        return feature, caption, quests, answers, scene_id, image_id


def visdial_collate_fn(data):
    """
    PAD = 0
    ? = 32525
    SOS = 32526
    EOS = 32527
    UNK = 32528
    """
    feature, captions, quests, answers, scene_id, image_id = zip(*data)

    new_feature = torch.tensor(list(feature))
    cap_max_len = np.max([len(cap) for cap in captions]) + 2
    que_max_len = np.max([[len(que) for que in ques] for ques in quests]) + 3
    ans_max_len = np.max([[len(ans) for ans in anses] for anses in answers]) + 2

    new_captions, new_quests, new_answers = [], [], []
    caption_msks, quest_msks, answer_msks = [], [], []
    for cap, quests, anses in zip(captions, quests, answers):
        new_cap = [32526] + cap + [32527] + [0 for _ in range(cap_max_len - len(cap) - 2)]
        cap_msk = [1. for _ in range(len(cap) + 2)] + [0. for _ in range(cap_max_len - len(cap) - 2)]
        new_captions.append(new_cap)
        caption_msks.append(cap_msk)
        new_quest, new_answer = [], []
        quest_msk, answer_msk = [], []
        for quest, ans in zip(quests, anses):
            new_que = [32526] + quest + [32525, 32527] + [0 for _ in range(que_max_len - len(quest) - 3)]
            new_ans = [32526] + ans + [32527] + [0 for _ in range(ans_max_len - len(ans) - 2)]
            que_msk = [1. for _ in range(len(quest) + 3)] + [0. for _ in range(que_max_len - len(quest) - 3)]
            ans_msk = [1. for _ in range(len(ans) + 2)] + [0. for _ in range(ans_max_len - len(ans) - 2)]
            new_quest.append(new_que)
            new_answer.append(new_ans)
            quest_msk.append(que_msk)
            answer_msk.append(ans_msk)
        new_quests.append(new_quest)
        new_answers.append(new_answer)
        quest_msks.append(quest_msk)
        answer_msks.append(answer_msk)

    new_captions = torch.tensor(new_captions)
    new_quests = torch.tensor(new_quests)
    new_answers = torch.tensor(new_answers)

    caption_msks = torch.tensor(caption_msks)
    quest_msks = torch.tensor(quest_msks)
    answer_msks = torch.tensor(answer_msks)

    entry = {
        'features': new_feature, # batch_size * 4096
        'captions': new_captions,# batch_size * 14
        'questions': new_quests, #  bs * 24
        'answers': new_answers, #  bs * 10 * 32
        'caption_msks': caption_msks, # bs * 17
        'question_msks': quest_msks, # bs * 10 * 21
        'answer_msks': answer_msks, # bs * 10 * 26
        'scene_id': list(scene_id), # int list, batch_size
        'image_id': list(image_id), # int list, batch_size
    }

    return entry


if __name__ == '__main__':
    train_loader = DataLoader(VisDialDataset(None, 'train'), collate_fn=visdial_collate_fn,
                              batch_size=4, shuffle=True, num_workers=4)
    print(len(VisDialDataset(None, 'val')))# 2064
    print(len(VisDialDataset(None, 'train')))# 123287
    for cnt, batched in enumerate(train_loader):
        print(batched["captions"])
    
        break
