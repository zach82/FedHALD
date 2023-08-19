#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from models import CNNFemnist
from concurrent.futures import ThreadPoolExecutor
import threading
import math
from losses import CosineSimilarityLoss, SupConLoss
import torch.nn.functional as F



class GlobalRepGenerator(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataloader = self.RAD_loader(dataset)

    def RAD_loader(self, dataset):
        """
        Returns a RAD_loader, I think the size of RAD should be a batch?
        """
        dataloader = DataLoader(dataset, batch_size=self.args.k,
                                  shuffle=True, drop_last=True)
        return dataloader

def cal_contrast_multithread(local_proto, global_protos, label_lists, tau):
    batch_num = local_proto.shape[0]
    cal_res = torch.Tensor([])
    for i in range(0, batch_num):
        flag = False
        denominator = 0
        numerator = 0
        for j in global_protos.keys():
            tmp = torch.exp(F.cosine_similarity(local_proto[i, :].unsqueeze(0), global_protos[j][0].data.unsqueeze(0)) / tau)
            denominator = denominator + tmp
            if j == label_lists[i].item():
                numerator = tmp
                flag = True
        if flag:
            cal_res = torch.cat((cal_res, torch.Tensor([torch.log(numerator / denominator) * -1])))
    return torch.mean(cal_res)

def cal_contrast_effective(local_proto, global_protos, label_lists, class_num, tau):
    bz = local_proto.shape[0]
    local_proto = local_proto.repeat_interleave(class_num, dim=0)
    indexes = []
    global_proto_list = []
    for i in range(class_num):
        global_proto_list.append(global_protos[i][0])
    global_protos = torch.stack(global_proto_list, dim=0)
    global_protos = global_protos.repeat((bz, ) + (1, ) * (global_protos.dim() - 1))
    cosine_similarities = F.cosine_similarity(local_proto, global_protos.detach(), dim=1)
    cosine_similarities = (cosine_similarities + 1) / 2
    cosine_similarities = torch.exp(cosine_similarities / tau)
    denominators = cosine_similarities.reshape(bz, class_num).sum(dim=1)
    for i in range(bz):
        indexes.append(label_lists[i].item() + i * class_num)
    numerators = cosine_similarities[indexes]
    res = -torch.log(numerators / denominators)
    return torch.mean(res)
    



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)
        self.criterion_cosin = CosineSimilarityLoss(args.lambda1)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights(self, idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_prox(self, idx, local_weights, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        if idx in local_weights.keys():
            w_old = local_weights[idx]
        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)
                if idx in local_weights.keys():
                    loss2 = 0
                    for para in w_avg.keys():
                        loss2 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss2 /= len(local_weights)
                    loss += loss2 * 150
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_het(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        # train_ep默认为1
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            # 用于告诉{自身标签：[每个样本的局部prototype,...,]}
            agg_protos_label = {}
            
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                # 前向传播得出一个batch的softmaxes和一个batch的prototypes
                log_probs, protos = model(images)
                # 求交叉熵loss1
                loss1 = self.criterion(log_probs, labels)

                # 均方误差loss_mse
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    #用i来指示目前为该batch中的第几个样本的prototype
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            #将全局的prototype赋值给proto_new
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    #对该batch中proto的均方误差
                    loss2 = loss_mse(proto_new, protos)
                
                #args.ld超参数正则项的权重
                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                #每个batch的准确率
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                # 不是每个batch都显示的
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
        epoch_loss['total'] = batch_loss['total'][-1]
        epoch_loss['1'] = batch_loss['1'][-1]
        epoch_loss['2'] = batch_loss['2'][-1]

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label


    def update_weights_het_v3(self, args, idx, global_protos, images_RAD, global_features, model, global_round=round):
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}
        agg_protos_label = {}
        agg_protos_label_num = {}
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)

                loss1 = self.criterion(log_probs, labels)

                if len(global_protos) == 0:
                    loss2 = loss1 * 0
                    loss3 = loss1 * 0
                else:
                    _, local_features = model(images_RAD)
                    loss2 = cal_contrast_effective(protos, global_protos, labels, self.args.num_classes, self.args.tau)
                    loss3 = self.criterion_cosin(local_features, global_features)
                    
                loss = loss1 + loss2 * self.args.ldc + loss3 * self.args.ld
                loss.backward()
                optimizer.step()
                
                log_probs = log_probs[:, 0: self.args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['3'].append(loss3.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['3'].append(sum(batch_loss['3']) / len(batch_loss['3']))

        # model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images, labels = images.to(self.device), label_g.to(self.device)
            model.zero_grad()
            _, protos = model(images)
            for i in range(len(labels)):
                if label_g[i].item() in agg_protos_label:
                    agg_protos_label[label_g[i].item()][0] += protos[i,:]
                    agg_protos_label_num[label_g[i].item()] += 1
                else:
                    agg_protos_label[label_g[i].item()] = [protos[i,:]]
                    agg_protos_label_num[label_g[i].item()] = 1
        for c in agg_protos_label.keys():
            agg_protos_label[label_g[i].item()][0] /= agg_protos_label_num[label_g[i].item()]
        
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['3'] = sum(epoch_loss['3']) / len(epoch_loss['3'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[: , 0 : args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = nn.NLLLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


def test_inference(args, model, test_dataset, global_protos):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs, protos = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(10):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc

def test_inference_new_cifar(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        cnt = np.zeros(100)
        for i in range(100):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(100):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct/total

    return acc

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):

    total, correct = 0.0, 0.0

    device = args.device
    # 所有客户端在测试集上使用神经网络和prototype的准确率加权
    acc_list_g = []
    # 所有客户端在测试集上使用神经网络的准确率
    acc_list_l = []
    # 所有客户端在测试集上prototype的LOSS
    loss_list = []
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        # 客户端i的最终准确率
        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

        total = 0
        correct = 0
        if global_protos != []:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs, protos = model(images)

                # compute the dist between protos and global_protos
                dist = torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                dist = dist * -1
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            dist[i, j] = F.cosine_similarity(protos[i, :].unsqueeze(0), global_protos[j][0].data.unsqueeze(0))

                # prediction
                _, pred_labels = torch.max(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)

    return acc_list_l, acc_list_g


def g_test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):

    total, correct1, correct2, correct3 = 0.0, 0.0, 0.0, 0.0

    device = args.device
    # 所有客户端在测试集上使用神经网络和prototype的准确率加权
    acc_list_g = []
    acc_list_m = []
    # 所有客户端在测试集上使用神经网络的准确率
    acc_list_l = []
    # 所有客户端在测试集上prototype的LOSS
    loss_list = []
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)
            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct1 += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            if global_protos != []:
                dist = torch.ones(size=(images.shape[0], args.num_classes)).to(device)
                dist = dist * -1
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            dist[i, j] = F.cosine_similarity(protos[i, :].unsqueeze(0), global_protos[j][0].data.unsqueeze(0))
                dist = (dist + 1) / 2
                
                _, pred_labels = torch.max(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct2 += torch.sum(torch.eq(pred_labels, labels)).item()

                # softmax_similarities = F.softmax(dist, dim=1)

                # softmax_similarities = args.lambda1 * torch.exp(outputs) + (1 - args.lambda1) * softmax_similarities
                # _, pred_labels = torch.max(softmax_similarities, 1)
                # pred_labels = pred_labels.view(-1)
                # correct3 += torch.sum(torch.eq(pred_labels, labels)).item()
                correct3 = 0

        # 客户端i的最终准确率
        acc1 = correct1 / total
        acc2 = correct2 / total
        acc3 = correct3 / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc1))
        acc_list_l.append(acc1)
        print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc2))
        acc_list_g.append(acc2)
        print('| User: {} | Global Test Acc with mixing protos: {:.5f}'.format(idx, acc3))
        acc_list_m.append(acc3)

    return acc_list_l, acc_list_g, acc_list_m

def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")

def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    dist = loss_mse(protos[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:,i], labels)).item()
        total += len(labels)

        cnt+=1
        if cnt==20:
            break

    acc = correct/total

    return acc