import pickle
import matplotlib.pyplot as plt
import numpy as np

rnn_0 = pickle.load(open('RNN-15-04-2024-13-37-44.p', 'rb'))
rnn_1 = pickle.load(open('RNN-15-04-2024-13-46-32.p', 'rb'))
rnn_2 = pickle.load(open('RNN-15-04-2024-13-55-12.p', 'rb'))
rnn_3 = pickle.load(open('RNN-15-04-2024-14-04-00.p', 'rb'))
rnn_4 = pickle.load(open('RNN-15-04-2024-14-12-50.p', 'rb'))
sns_0 = pickle.load(open('SNS-15-04-2024-11-34-46.p', 'rb'))
sns_1 = pickle.load(open('SNS-15-04-2024-11-57-16.p', 'rb'))
sns_2 = pickle.load(open('SNS-15-04-2024-12-19-53.p', 'rb'))
sns_3 = pickle.load(open('SNS-15-04-2024-12-42-04.p', 'rb'))
sns_4 = pickle.load(open('SNS-15-04-2024-13-04-15.p', 'rb'))

max_len = 51
def plot_data(data, label, color0, color1):
    loss = np.zeros([5,max_len])
    acc_train = np.zeros_like(loss)
    acc_test = np.zeros_like(loss)
    for i in range(len(data)):
        loss[i,:] = data[i]['loss'][:max_len]
        acc_train[i,:] = data[i]['accTrain'][:max_len]
        acc_test[i,:] = data[i]['accTest'][:max_len]
    loss_mean = np.mean(loss, axis=0)
    loss_5 = np.percentile(loss, 5, axis=0)
    loss_95 = np.percentile(loss, 95, axis=0)
    acc_train_mean = np.mean(acc_train, axis=0)
    acc_train_5 = np.percentile(acc_train, 5, axis=0)
    acc_train_95 = np.percentile(acc_train, 95, axis=0)
    acc_test_mean = np.mean(acc_test, axis=0)
    acc_test_5 = np.percentile(acc_test, 5, axis=0)
    acc_test_95 = np.percentile(acc_test, 95, axis=0)
    x = np.array(range(max_len))

    plt.subplot(1,2,1)
    plt.fill_between(x, loss_95, loss_5, color=color0, alpha=0.5)
    plt.plot(x, loss_mean, color=color0, label=label)
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.subplot(1,2,2)
    plt.fill_between(x, acc_train_95, acc_train_5, color=color0, alpha=0.5)
    plt.plot(x, acc_train_mean, color=color0, label=label+' Training Accuracy')
    plt.fill_between(x, acc_test_95, acc_test_5, color=color1, alpha=0.5)
    plt.plot(x, acc_test_mean, color=color1, label=label+' Test Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('% Accuracy')

plt.figure()
plot_data([rnn_0,rnn_1,rnn_2,rnn_3,rnn_4], 'RNN', 'C0', 'C1')
plot_data([sns_0,sns_1,sns_2,sns_3,sns_4], 'SNS', 'C2', 'C3')
plt.show()
