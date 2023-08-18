import json
import tkinter as tk
import torch
import numpy as np
from model import *
def couplet_match(s, device='cpu'):
    # 将字符串转为数值
    with open('./save/word2idx.json', 'r') as f:
        word2idx = json.load(f)
    with open('./save/idx2word.json', 'r') as f:
        idx2word = json.load(f)
    model_path = './save/best_model.pt'
    model = torch.load(model_path)
    model.to(device)
    x = [word2idx[word] for word in s]

    # 将数值向量转为tensor
    x = torch.from_numpy(np.array(x).reshape(-1, 1))

    y = model(x)
    y = y.argmax(axis=1)
    r = ''.join([idx2word.get(str(idx.item())) for idx in y])

    print('上联：%s，下联：%s' % (s, r))
    return r


def insert_point():
    var = e.get('0.0', 'end')
    var = var.strip()
    # print('hhhh',var,type(var))
    second = couplet_match(var)
    # ee.insert('end',str(tensor))
    t.delete('1.0', 'end')
    t.insert('end', second)


def delet():
    e.delete('1.0', 'end')

window = tk.Tk()
window.title('对联生成器')
window.geometry('600x400')
b1 = tk.Button(window, text='点我', width=10,
               height=2, command=insert_point)
b1.place(x=200,y=150)

b2 = tk.Button(window, text='清空输入~', width=10,
               height=2, command=delet)
b2.place(x=300,y=150)

e = tk.Text(window,width=57,height=5,font=('楷体', 15), show = None)#显示成明文形式
e.place(x=10,y=10)

t = tk.Text(window,width=57, height=5,font=('楷体', 15))
t.place(x=10,y=220)

window.mainloop()