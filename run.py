import numpy as np
import torch
from torch import nn
from torch import optim
from torchnet import meter
from dataprocessor import *
from model import *
import os
import matplotlib.pyplot as plt

# 模型输入参数，需要自己根据需要调整
input_path = './data/train_in.txt'
output_path = './data/train_out.txt'
test_input_path = './data/test_in.txt'
test_output_path = './data/test_out.txt'

num_layers = 3  # LSTM层数
hidden_dim = 512  # LSTM中的隐层大小
epochs = 150  # 迭代次数
batch_size = 256  # 每个批次样本大小
embedding_dim = 128  # 每个字形成的嵌入向量大小
lr = 0.001 # 学习率
device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dict, vocab_size, idx2word, word2idx, max_len = load_data(input_path, output_path)
# 模型训练
model = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim,
             embedding_dim=embedding_dim, num_layers=num_layers)

best_model = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim,
             embedding_dim=embedding_dim, num_layers=num_layers)


optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
model.to(device)
loss_meter = meter.AverageValueMeter()
best_loss = float('inf')
val_loss_meter = meter.AverageValueMeter()

loss_arr = []
val_loss_arr = []

for epoch in range(epochs):
    loss_meter.reset()
    val_loss_meter.reset()
    for x, y in data_generator(train_dict, batch_size=batch_size, max_len=max_len):
        model.train()
        idxs = np.random.choice(len(x), size=max(int(len(x)*0.1), 1))

        valid_x = torch.from_numpy(x.copy()[idxs]).long().transpose(1, 0).contiguous()
        valid_y = torch.from_numpy(y.copy()[idxs]).long().transpose(1, 0).contiguous()
        valid_x = valid_x.to(device)
        valid_y = valid_y.to(device)

        train_x = torch.from_numpy(np.delete(x, idxs, 0)).long().transpose(1, 0).contiguous()
        train_x = train_x.to(device)

        train_y = torch.from_numpy(np.delete(y, idxs, 0)).long().transpose(1, 0).contiguous()
        train_y = train_y.to(device)

        optimizer.zero_grad()

        output_ = model(train_x)

        loss = criterion(output_, train_y.long().view(-1))
        loss.backward()

        optimizer.step()
        loss_meter.add(loss.item())
        model.eval()
        # valid_x = train_x[idxs]
        # valid_y = train_y[idxs]
        try:
            output_ = model(valid_x)
        except:
            print(111)
        val_loss = criterion(output_, valid_y.long().view(-1))
        val_loss_meter.add(val_loss.item())

    # 打印信息
    print("【EPOCH:     】%s" % str(epoch + 1))
    print("【Train Loss:】%s" % (str(loss_meter.mean)))
    print("【Valid Loss:】%s" % (str(val_loss_meter.mean)))
    loss_arr.append(loss_meter.mean)
    val_loss_arr.append(val_loss_meter.mean)
    # 保存模型及相关信息
    if val_loss_meter.mean < best_loss:
        best_loss = val_loss_meter.mean
        best_model.load_state_dict(model.state_dict())

    # 在训练结束保存最优的模型参数
    if epoch == epochs - 1:
        # 保存模型
        torch.save(best_model,
                   './save/best_model.pt')
plt.plot(loss_arr, label='loss')
plt.plot(val_loss_arr, label='val_loss')
plt.legend()
plt.savefig('./save/loss.png')

test_dict = load_test_data(test_input_path, test_output_path, word2idx)
best_model.to(device)
best_model.eval()
loss_meter.reset()
for x, y in data_generator(test_dict, batch_size=batch_size, max_len=max_len):
    x = torch.from_numpy(x).long().transpose(1, 0).contiguous()
    x = x.to(device)
    y = torch.from_numpy(y).long().transpose(1, 0).contiguous()
    y = y.to(device)
    output_ = best_model(x)
    loss = criterion(output_, y.long().view(-1))
    loss_meter.add(loss.item())

print("【Test Loss:】%s" % (str(loss_meter.mean)))
