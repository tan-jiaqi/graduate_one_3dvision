from __future__ import print_function
import time
import wandb

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Sequence(nn.Module):
    def __init__(self, num_layers=2, hidden_size=51):
        super(Sequence, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(1 if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, future=0, device='cpu'):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double, device=device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double, device=device)
        # h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=device)
        # c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm_cells[0](input_t, (h_t, c_t))
            for i in range(1, len(self.lstm_cells)):
                h_t, c_t = self.lstm_cells[i](h_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):  # 预测时间序列
            h_t, c_t = self.lstm_cells[0](output, (h_t, c_t))
            for j in range(1, len(self.lstm_cells)):
                h_t, c_t = self.lstm_cells[j](h_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    parser.add_argument('--lr', type=float, default=0.8, help='learning rate')
    parser.add_argument("--layers", type=int, default=2, help="number of layers")
    parser.add_argument('--hidden_size', type=int, default=51, help='hidden size')
    parser.add_argument('--device', type=str, default='cuda', help='device to run')
    parser.add_argument('--mode', type=str, default='TrainAndTest', help='mode to run')
    opt = parser.parse_args()
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    print('Using device:', device)
    # 设置初始seed
    np.random.seed(0)
    torch.manual_seed(0)
    # 加载数据
    data = torch.load('datasets/traindata.pt')
    input = torch.from_numpy(data[3:, :-1]).to(device)
    target = torch.from_numpy(data[3:, 1:]).to(device)
    test_input = torch.from_numpy(data[:3, :-1]).to(device)
    test_target = torch.from_numpy(data[:3, 1:]).to(device)
    # 创建模型
    seq = Sequence(num_layers=opt.layers, hidden_size=opt.hidden_size).to(device)
    seq.double()
    criterion = nn.MSELoss()
    # 使用LBFGS优化器
    optimizer = optim.LBFGS(seq.parameters(), lr=opt.lr)
    if (opt.mode == 'TrainAndTest'):
        # 设置wandb
        wandb.init(project='lstm-sine-wave', name=time.strftime('%Y-%m-%d %H:%M:%S'),
                   config={"steps": opt.steps,
                           "lr": opt.lr,
                           "hidden size": opt.hidden_size,
                           "lstm cells number": opt.layers,
                           "mode": opt.mode})
        # 训练日志
        train_log = {}
        # 训练模型
        for i in range(opt.steps):
            print('STEP: ', i)
            # 自定义优化器
            def closure():
                optimizer.zero_grad()
                out = seq(input, device=device)
                loss = criterion(out, target)
                print('loss:', loss.item())
                loss.backward()
                return loss
            # 自动进行优化
            optimizer.step(closure)
            # 测试loss
            with torch.no_grad():
                future = 1000
                pred = seq(test_input, future=future, device=device)
                loss = criterion(pred[:, :-future], test_target)
                print('test loss:', loss.item())
                train_log['test_loss'] = loss.item()
                wandb.log(train_log)
                y = pred.detach().cpu().numpy()  # 将预测结果转换为numpy数组

            # 画图
            plt.figure(figsize=(30, 10))
            plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)


            def draw(yi, color):
                plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
                plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':',
                         linewidth=2.0)


            draw(y[0], 'r')
            draw(y[1], 'g')
            draw(y[2], 'b')
            plt.savefig('predictions/predict_%dlayers_%dhsize_%depochs.jpg' % (opt.layers, opt.hidden_size, i))
            plt.close()

        # 保存模型
        torch.save(seq.state_dict(), 'model/lstm_model.pth')
        wandb.finish()
    elif (opt.mode == 'Validate'):
        # 验证模型
        seq.load_state_dict(torch.load('model/lstm_model.pth'))
        seq.eval()
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future, device=device)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().cpu().numpy()

        # 画图
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predictions/predict_%dlayers_%dhsize_final.jpg' % (opt.layers, opt.hidden_size))
        plt.close()


