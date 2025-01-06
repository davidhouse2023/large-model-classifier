import logging
import os
from datetime import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix, f1_score
from tqdm import tqdm
import seaborn as sns
import torch.distributed as dist


def evaluate_model(model, dataloader, rank, mlflow):
    model.eval()
    all_preds_code = []
    all_code_labels = []
    all_preds_complexity = []
    all_complexity_labels = []
    total_loss = 0

    with torch.no_grad():
        # for batch in tqdm(dataloader, desc=f"validation", file=open("./log/training.log", "a")):
        for index, batch in enumerate(tqdm(dataloader, desc=f"validation")):
            input_ids = batch['input_ids'].cuda(rank)
            attention_mask = batch['attention_mask'].cuda(rank)
            labels = batch["labels"].cuda(rank)

            # 向模型前向传播
            outputs, loss = model(input_ids, attention_mask, labels)
            loss = loss.sum()

            total_loss += loss.item()

            # 对code类别的概率使用sigmoid，并根据阈值0.5判断类别
            probs_code = torch.sigmoid(outputs[:, 0])
            preds_code = (probs_code > 0.5).float()

            # 难易程度
            _, preds_complexity = torch.max(outputs[:,1:], dim=1)
            _, true_labels = torch.max(labels[:,1:], dim=1)

            # 收集预测值和真实标签
            all_preds_code.extend(preds_code.cpu().numpy())
            all_code_labels.extend(labels[:,0].cpu().numpy())
            all_complexity_labels.extend(true_labels.cpu().numpy())
            all_preds_complexity.extend(preds_complexity.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    complexity_accuracy = accuracy_score(all_complexity_labels, all_preds_complexity)
    code_accuracy = accuracy_score(all_code_labels, all_preds_code)
    complexity_f1_macro = f1_score(all_complexity_labels,all_preds_complexity, average='weighted')

    mlflow.log_metric('Code Accuracy', code_accuracy)
    mlflow.log_metric('Complexity Accuracy', complexity_accuracy)
    mlflow.log_metric('Complexity F1 Score', complexity_f1_macro)
    logging.info(f'Code Accuracy: {code_accuracy:.4f}')
    logging.info(f'Complexity Accuracy: {complexity_accuracy:.4f}')
    logging.info(f'Complexity F1 Score: {complexity_f1_macro:.4f}')

    # 加权平均
    overall_accuracy = (code_accuracy * 0.3 + complexity_accuracy * 0.7)
    mlflow.log_metric('overall_accuracy', overall_accuracy)

    # 生成混淆矩阵
    conf_mat_complexity = confusion_matrix(all_complexity_labels, all_preds_complexity)
    conf_mat_code = confusion_matrix(all_code_labels, all_preds_code)

    # return avg_loss, overall_accuracy, conf_mat, conf_mat_code
    return avg_loss, overall_accuracy,complexity_accuracy,complexity_f1_macro,conf_mat_complexity,conf_mat_code


def train_model(model, train_dataloader, eval_dataloader, optimizer, scheduler, rank, mlflow, layers ,num_epochs=3):
    model.train()
    accuracies = []
    best_confusion_matrices = None
    losses = []
    batch_losses = [0]
    val_losses = []
    best_acc = 0
    best_model = None
    step=0

    for epoch in range(num_epochs):
        start = time.time()
        epoch_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
            step+=1
            input_ids = batch['input_ids'].cuda(rank)
            attention_mask = batch['attention_mask'].cuda(rank)
            labels = batch['labels'].cuda(rank)
            # 清空优化器
            optimizer.zero_grad()
            # 向模型前向传播
            outputs, loss = model(input_ids, attention_mask, labels)
            loss = loss.sum()

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 计算损失
            epoch_loss += loss.item()
            batch_losses.append(loss.item())
            logging.info(batch_losses[-1])
            mlflow.log_metric('loss', loss.item(), step=batch_idx)

            # 每1000个batch评估一次
            if (batch_idx + 1) % 1000 == 0:

                logging.info(f"Evaluating model at batch {batch_idx + 1} of epoch {epoch + 1}")

                val_loss, overall_accuracy,complexity_accuracy,complexity_f1_macro,conf_mat_complexity,conf_mat_code = evaluate_model(model, eval_dataloader, rank,mlflow)

                val_losses.append(val_loss)
                accuracies.append(overall_accuracy)

                # if overall_accuracy > best_acc:
                #     best_acc = overall_accuracy
                #     best_confusion_matrices = [conf_mat_complexity,conf_mat_code]
                #     best_model = model.state_dict()
                #     logging.info(f'当加权平均得到的指标best_acc：{best_acc}最好时，其他各项指标为：Tcomplexity_accuracy: {complexity_accuracy:.4f},complexity_f1_macro: {complexity_f1_macro:.4f}')
                if rank==0:
                    torch.save(model.module.state_dict(), f"./result/qwen/DDP/best_model_layer{layers}_step{step}_acc_{complexity_accuracy}.pt")
                    logging.info(f"best_model_layer{layers}_step{step}_acc_{complexity_accuracy}.pt保存成功！")

                # 记录每1000个batch的平均损失
                avg_batch_loss = sum(batch_losses) / len(batch_losses)
                logging.info(f'Train Loss: {avg_batch_loss:.4f},Val Loss: {val_loss:.4f}')
                losses.append(avg_batch_loss)
                batch_losses = []  # 重置batch_losses

        end = time.time()
        logging.info("epoch:{:3d} use time: {:.4f}".format(epoch, end - start))
        now = datetime.now()
        formatted_now = now.strftime("%Y%m%d%H%M%S")
        # torch.save(best_model, f"./result/qwen/best_model_layer_{layers}_{formatted_now}.pt")

    logging.info("Finished Training")
    dist.destroy_process_group()

    if not os.path.exists(f"./result/{formatted_now}"):
        os.makedirs(f"./result/{formatted_now}")

    # 绘制损失变化图
    plt.plot(range(1000, (len(losses) + 1) * 1000, 1000), losses, label='Training Loss', color='blue', marker='o',
             linestyle='-')
    plt.plot(range(1000, (len(val_losses) + 1) * 1000, 1000), val_losses, label='Validation Loss', color='red',
             marker='o', linestyle='-')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.savefig(f"./result/{formatted_now}/loss_plot.png")
    plt.show()

    # 绘制acc曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1000, (len(accuracies) + 1) * 1000, 1000), accuracies, marker='o', linestyle='-')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('accuracy per 1000 Batches on validation set')
    plt.savefig(f"./result/{formatted_now}/accuracy_plot.png")
    plt.show()

    # # 类别名称
    # class_names =["very easy", "easy", "medium", "hard", "very hard"]  # 根据你的实际标签类别进行修改

    # # 使用 Seaborn 绘制混淆矩阵
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(best_confusion_matrices[0], annot=True, fmt='d', cmap='Blues',
    #             xticklabels=class_names,
    #             yticklabels=class_names)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Complexity Confusion Matrix')
    # plt.savefig(f"./result/{formatted_now}/Complexity_Confusion_Matrix.png")
    # plt.show()
    #
    # # 绘制混淆矩阵热力图
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(best_confusion_matrices[1], annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Code Complexity Confusion Matrix')
    # plt.savefig(f"./result/{formatted_now}/Code_Confusion_Matrix.png")
    # plt.show()
