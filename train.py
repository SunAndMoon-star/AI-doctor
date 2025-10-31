from parameter_config import ParameterConfig
from transformers import BertTokenizerFast,GPT2LMHeadModel,GPT2Config
from utils.functions_tools import calculate_accuracy
from demo_01 import get_dataloader
import transformers
from torch.optim import AdamW
import torch
import os
from tqdm import tqdm

def train_epoch(model,train_dataloder,optimizer,lr_scheduler,epoch,params):
    model.train()
    total_loss = 0
    loss_list=[]
    epoch_correct_num=0
    epoch_total_word=0
    for batch_idx,(input_id,label) in enumerate(tqdm(train_dataloder),start=1):
        input_ids = input_id.to(params.device)
        labels=label.to(params.device)
        outputs=model(input_ids=input_ids,labels=labels)
        logits=outputs.logits
        loss=outputs.loss
        loss=loss.mean()
        loss_list.append(loss.item())
        total_loss += loss.item()

        n_correct,n_word=calculate_accuracy(logits,labels,ignore_index=params.ignore_index)
        #计算训练批次准确率
        batch_accuracy=n_correct/n_word
        epoch_correct_num+=n_correct
        epoch_total_word+=n_word
        if params.gradient_accumulation_steps > 1:
            loss=loss/params.gradient_accumulation_steps

        #反向传播
        loss.backward()
        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(),params.max_grad_norm)

        #达到一定步数之后进行参数更新
        if batch_idx % params.gradient_accumulation_steps ==0:
            #更新参数
            optimizer.step()
            #更新学习率
            lr_scheduler.step()
            #梯度清零
            optimizer.zero_grad()

        #日志打印
        if batch_idx %params.loss_step ==0:
            ture_loss=loss*params.gradient_accumulation_steps
            lr=lr_scheduler.get_last_lr()[0]
            print('Epoch:{}|Loss:{:.5f}|Train_Accuracy:{:.5f}|Lr{:.5f}'.format(epoch,ture_loss,batch_accuracy,lr))

        #记录当前epoch的平均loss
        epoch_mean_loss=total_loss/len(train_dataloder)
        epoch_mean_acc=epoch_correct_num/epoch_total_word
        print('平均loss:{}|平均准确率{}'.format(epoch_mean_loss,epoch_mean_acc))

    if epoch%10==0 or epoch== params.epochs:
        model_path=os.path.join(params.save_model_path,'epoch_{}'.format(epoch))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)

    return epoch_mean_loss
def validate_epoch(model,valid_dataloder,epoch,params):
    total_loss=0
    loss_list=[]
    epoch_correct_num = 0#统计当前验证集epoch的准确数
    epoch_total_word = 0#统计当前验证集epoch的词总数数
    #将模型设置为评估
    model.eval()
    with torch.no_grad():
        for batch_idx,(input_id,labels) in enumerate(tqdm(valid_dataloder),start=1):
            input_id=input_id.to(params.device)
            labels=labels.to(params.device)
            #模型预测
            outputs=model(input_id=input_id,labels=labels)
            logits=outputs.logits
            loss=outputs.loss
            loss_list.append(loss.item())
            loss=loss.mean()
            total_loss+=loss.item()
            batch_n_correct,batch_n_word=calculate_accuracy(logits,labels,ignore_index=params.ignore_index)
            #训练批次准确率
            # batch_accuracy=batch_n_correct/batch_n_word
            epoch_correct_num+=batch_n_correct
            epoch_total_word+=batch_n_word
    #计算验证集平均loss和准确率
    val_mean_loss=total_loss/len(valid_dataloder)
    val_mean_acc=epoch_correct_num/epoch_total_word
    print('Epoch: {}/{}|val_mean_loss:{:.5f}|val_mean_acc:{:.5f}'.format(epoch,val_mean_loss,val_mean_acc))


    return val_mean_loss
def train(model,params):
    train_dataloder,valid_dataloder=get_dataloader(params.train_path,params.valid_path,params.batch_size)
    print(params.batch_size)


    #优化器
    optim=AdamW(model.parameters(),lr=params.lr,eps=params.eps)
    t_total=len(train_dataloder)//params.gradient_accumulation_steps*params.epochs
    lr_scheduler=transformers.get_linear_schedule_with_warmup(optim,num_warmup_steps=params.num_warmup_steps,num_training_steps=t_total)
    #遍历论述
    best_valid=100000
    train_losses=[]
    valid_losses=[]
    for epoch in range(1,params.epochs+1):

        #模型训练
        train_loss=train_epoch(model,train_dataloder,optim,lr_scheduler,epoch,params)
        train_losses.append(train_loss)
       #模型评估
        valid_loss=validate_epoch(model,valid_dataloder,epoch,params)
        valid_losses.append(valid_loss)
         #模型保存:保存当前准确率高，损失低的模型
        if valid_loss<best_valid:
            best_valid_loss=valid_loss
            model_path=os.path.join(params.save_model_path,'min_loss_model_{}'.format(epoch))
            # print(model_path)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model.save_pretrained(model_path)
        torch.cuda.empty_cache()  # 每个epoch结束后调用


def main():

    #加载配置参数
    params=ParameterConfig()
    #初始化tokenizer
    tokenizer = BertTokenizerFast(params.vocab_path,
                                  sep_token="[SEP]",
                                  pad_token="[PAD]",
                                  cls_token="[CLS]")
    # print('tokenizer.vocab_size:', tokenizer.vocab_size)
    sep_id = tokenizer.sep_token_id  # 获取分隔符[SEP]的token ID
    cls_id = tokenizer.cls_token_id  # 获取起始符[CLS]的token ID
    # print(sep_id)
    #初始化模型
    # 加载预训练模型
    if params.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    # 初始化模型
    else:
        model_config = GPT2Config.from_json_file(params.config_json)
        model = GPT2LMHeadModel(config=model_config)
    model=model.to(params.device)
    # print(model)
    #断言：确认vocab_size一致
    print('tokenizer.vocab_size:', tokenizer.vocab_size)
    print('model.config.vocab_size:', model.config.vocab_size)
    assert model.config.vocab_size == tokenizer.vocab_size
    #创建模型的保存目录
    if not os.path.exists(params.save_model_path):
        os.mkdir(params.save_model_path)

    #模型训练（训练+评估）
    train(model,params)

if __name__ == '__main__':
    main()