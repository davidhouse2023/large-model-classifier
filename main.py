# -*- coding: utf-8 -*-
import logging
import mlflow

from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, get_scheduler, SchedulerType

from dataset import InstructionDataset, myDataset
from model import LlamaClassifier, Qwen25Classifier
from trainer import train_model

mlflow.set_tracking_uri("http://")
mlflow.set_experiment("Qwen25Classifier_20240929")
# mlflow.create_experiment("LlamaClassifier")
mlflow.start_run(run_name='8_layers')
# 设置日志格式
log_format = '%(asctime)s - %(levelname)s - %(message)s'

# 设置日志基本配置
logging.basicConfig(
    filename='./logs/qwen/training_code_8.log',  # 日志文件路径
    filemode='w',  # 写入模式（覆盖）
    format=log_format,  # 日志格式
    level=logging.INFO  # 日志等级
)

# 创建控制台处理器并设置格式
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设置控制台日志级别
console_handler.setFormatter(logging.Formatter(log_format))  # 设置控制台日志格式

# 获取根日志器并添加控制台处理器
logger = logging.getLogger()
logger.addHandler(console_handler)

# MODEL_PATH = "/data4/llamaBR_Small_8b_funcall/hf_model"
MODEL_PATH = '/opt/ailab_mnt1/LLM_MODELS/LLAMA/Qwen2.5_7B_Instruct'

if __name__ == '__main__':

    # path = "/ailab_mnt/weili.zhang/llm_classifier/instruction.jsonl"
    train_path="./data/instruction.jsonl"
    valid_path="./data/valid_data_code_zh.jsonl"
    batch_size = 32
    num_classes = 6
    lr = 1e-3
    model_layer = 8

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info("create dataset...")
    train_dataset = InstructionDataset(train_path, tokenizer,split="train")
    logging.info(f"train_dataset:{len(train_dataset)}")

    val_dataset = InstructionDataset(valid_path, tokenizer,split="val")
    logging.info(f"val_dataset:{len(val_dataset)}")
    logging.info("create DataLoader...")


    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # model = LlamaClassifier(MODEL_PATH, num_classes=num_classes, layer_index=model_layer, tokenizer=tokenizer).to(
    #     dtype=torch.bfloat16)
    model = Qwen25Classifier(MODEL_PATH, num_classes=num_classes, layer_index=model_layer, tokenizer=tokenizer).to(
        dtype=torch.bfloat16)

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[6, 7])

    model.to(device)
    # 定义优化器和损失函数
    params_to_update = []
    for name, param in model.named_parameters():
        if 'score' in name:
            params_to_update.append(param)

    optimizer = optim.AdamW(params_to_update, lr=lr)
    num_epochs = 2
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_rate = 0.01
    scheduler = get_scheduler(SchedulerType.COSINE,
                              optimizer=optimizer,
                              num_warmup_steps=int(num_training_steps * num_warmup_rate),
                              num_training_steps=num_training_steps)

    logging.info(f"batch_size:{batch_size},lr:{lr},model layer:{model_layer},num_epochs：{num_epochs}")

    # 开始训练
    logging.info("start train...")
    train_model(model, train_loader, val_loader, optimizer, scheduler, device, mlflow, model_layer, num_epochs)
    mlflow.end_run()
