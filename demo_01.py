from datasets import Dataset
from transformers import BertTokenizerFast#分词工具
import pickle as pkl
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn.utils.rnn as rnn_utils


def data_preprocess(txt_path,pkl_path):
    tokenizer = BertTokenizerFast('./vocab/vocab.txt',
                                  sep_token="[SEP]",
                                  pad_token="[PAD]",
                                  cls_token="[CLS]")
    # print('tokenizer.vocab_size:', tokenizer.vocab_size)
    sep_id = tokenizer.sep_token_id  # 获取分隔符[SEP]的token ID
    cls_id = tokenizer.cls_token_id  # 获取起始符[CLS]的token ID
    #读取训练数据集
    with open(txt_path,'rb') as f:
        data=f.read().decode('utf-8')
    if '\r\n' in data:
        train_data=data.split('\r\n\r\n')
    else:
        train_data=data.split('\n\n')

    dialogue_list=[]
    for index,dialogue in enumerate(tqdm(train_data)):
        if '\r\n' in dialogue:
            sequences=dialogue.split('\r\n')
        else:
            sequences=dialogue.split('\n')
        input_ids=[cls_id]
        for sequence in sequences:
            encode_seq1=tokenizer.encode(sequence,add_special_tokens=False)
            input_ids+=encode_seq1
            input_ids.append(sep_id)
        dialogue_list.append(input_ids)
    with open(pkl_path,'wb') as f:
        pkl.dump(dialogue_list,f)



class MyDataset(Dataset):
    def __init__(self,input_list,max_len):
        super(MyDataset, self).__init__()
        self.input_list = input_list  # 样本id
        self.max_len = max_len  # 样本长度
    def __len__(self):
        return len(self.input_list)
    def __getitem__(self,index):
        input_ids = self.input_list[index]# 获取给定索引处的输入序列
        input_ids = input_ids[:self.max_len]  # 根据最大序列长度对输入进行截断
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids


#获取dataset
def load_dataset(train_path,valid_path):
    with open(train_path,'rb') as f:
        train_input_list = pkl.load(f)
    with open(valid_path, "rb") as f:
        valid_input_list = pkl.load(f)
    train_dataset = MyDataset(input_list=train_input_list, max_len=300)
    valid_dataset = MyDataset(input_list=valid_input_list, max_len=300)
    return train_dataset, valid_dataset


#回调函数
def collate_fn(data):
    input_ids = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(data, batch_first=True, padding_value=-100)
    return input_ids, labels


def get_dataloader(train_path,valid_path,batch_size=4):
    train_dataset, valid_dataset = load_dataset(train_path,valid_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True, collate_fn=collate_fn,drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size,shuffle=True, collate_fn=collate_fn,drop_last=True)
    return train_dataloader, valid_dataloader
if __name__ == '__main__':
    # txt_path = './data/medical_train.txt'
    txt_path = './data/medical_valid.txt'
    pkl_path = './data/medical_valid.pkl'
    data_preprocess(txt_path, pkl_path)