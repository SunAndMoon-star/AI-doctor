import torch
class ParameterConfig():
    def __init__(self):
        # 判断是否使用GPU（1.电脑里必须有显卡；2.必须安装cuda版本的pytorch）
        self.device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
        print('Using Device:{}'.format(self.device))
        # 词典路径：在vocab文件夹里面
        self.vocab_path = './vocab/vocab.txt'
        # 训练数据文件路径
        self.train_path = './data/medical_train.pkl'
        # 验证数据文件路径
        self.valid_path = './data/medical_valid.pkl'
        # 模型配置文件
        self.config_json = './config/config.json'
        self.pretrained_model = ''
        self.batch_size = 16
        self.lr=2.6e-5
        self.eps=1.0e-9
        self.epochs=50
        self.save_model_path='./save_model'
        self.loss_step=20
        self.gradient_accumulation_steps=4
        self.max_grad_norm=2.0
        self.num_warmup_steps=50
        self.ignore_index=-100#忽略字符
