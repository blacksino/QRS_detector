import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import os
from datetime import datetime
import torch
import torch.nn
import tqdm
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from Dataset.noisyR_peak import NoisyR
from model.BiLSTM_QRSdetector import BiLSTM
from torchmetrics.functional import accuracy

parser = argparse.ArgumentParser(description='Robust_ECG_QRSDetection')

parser.add_argument('--db_path', required=False)
parser.add_argument('--log_path', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--win_size', default=1000)
parser.add_argument('--load_model', default=False)

args = parser.parse_args()
learning_rate = 1e-4
device = torch.device("cuda")
batch_size = 256
epochs = 150
gpus = [0, 1]


def main():
    dataset = NoisyR(args, db_name='mitdb', channel=0, from_scratch=False)
    print("Dataset merge done.")

    MyNet = BiLSTM(args.win_size, n_hidden=64, n_layer=2)
    MyNet = MyNet.cuda()
    MyNet = torch.nn.DataParallel(MyNet, device_ids=gpus)
    print('the model has been sent to GPU')

    if args.load_model:
        load_model(args.model_path, MyNet)

    optimizer = torch.optim.Adam(MyNet.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(epochs):
        MyNet.train()
        print('Epoch', epoch)
        tq = tqdm.tqdm(total=len(dataloader)*batch_size, dynamic_ncols=True, ncols=100)
        for batch_idx, data in enumerate(dataloader):
            signal = data[0].float().cuda().reshape(batch_size, -1, 1)
            label = data[1].float().cuda()

            predict = MyNet(signal).squeeze()
            loss = criterion(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.update(batch_size)
            tq.set_postfix(
                loss='cur: {:.5f} '.format(loss.item()))
        tq.close()
        acc = validate(dataloader, MyNet)
        print(f'Current acc:{acc}')
    save_model(MyNet, epoch)


@torch.no_grad()
def validate(dataloader, MyNet):
    MyNet.eval()
    total_acc = 0
    iters = 0
    for batch_idx, data in enumerate(dataloader):

        signal = data[0].float().cuda().reshape(batch_size, -1, 1)
        labels = data[1].long().cuda()

        pred = MyNet(signal)
        total_acc += accuracy(pred.squeeze(), labels)
        iters+=1

    return total_acc/iters


def save_model(model, epoch):
    currentDT = datetime.now()
    log_root = f'{args.model_path}/{currentDT.month}_{currentDT.day}_{currentDT.hour}/'

    if not os.path.exists(log_root):
        os.mkdir(log_root)

    path = os.path.join(log_root, f'QRS_best_score_at_epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict()
    }, path)
    print(f'Current model has been saved.This is Epoch{epoch}.')


def load_model(path, model):
    print('loading model....')
    state_dict = torch.load(path)
    epoch = state_dict['epoch']
    model.load_state_dict(state_dict['state_dict'])
    print(f'last time we are on {epoch}.')
    print('The model has been loaded.')


if __name__ == '__main__':
    main()
