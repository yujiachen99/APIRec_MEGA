import numpy as np
import torch
import torch.nn as nn 
from model import MEGA
import logging
from tqdm import tqdm
from collections import defaultdict
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

 
def train(args, data_info):

    logging.info("================== training MEGA ====================")
    train_data = data_info[0] 
    train_data = np.array(train_data) 
    test_data = data_info[1] 
    test_data = np.array(test_data) 
    user_triple_set = data_info[4] 
    user_triple_set_e = data_info[6] 
    item_triple_set = data_info[5] 
    item_triple_set_i = data_info[7] 
    model, optimizer, loss_func = _init_model(args, data_info)
    for step in range(args.n_epoch):
        np.random.shuffle(train_data)
        start = 0 
        epoch_loss = 0 
        while start < train_data.shape[0]:
            labels = _get_feed_label(args, train_data[start:start + args.batch_size, 2])
            scores = model(*_get_feed_data(args, train_data, user_triple_set,user_triple_set_e, item_triple_set,item_triple_set_i, start, start + args.batch_size))
            loss = loss_func(scores, labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += args.batch_size
        s = 'epoch: {} loss:{}'.format(step, epoch_loss)
        logging.info(s)   
        if (step + 1) % 1 == 0:
            logging.info("================== testing MEGA ====================")   
            topk_eval(args, model, train_data, test_data,  user_triple_set,user_triple_set_e, item_triple_set,item_triple_set_i)


def topk_eval(args, model, train_data, test_data, user_triple_set,user_triple_set_e, item_triple_set,item_triple_set_i):
    # logging.info('calculating recall ...')
    f = open('./result.txt', 'w')
    k_list = [1, 5, 10, 20]

    success_list = {k: [] for k in k_list}
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())
    train_record = _get_user_record(args, train_data, True)
    test_record = _get_user_record(args, test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))

    user_list = set(test_data[:,0])
    model.eval()
    
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size] 
            input_data = _get_topk_feed_data(user, items)
            scores = model(*_get_feed_data(args, input_data,user_triple_set, user_triple_set_e, item_triple_set, item_triple_set_i, 0, args.batch_size)).cpu().detach().numpy()
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            scores = model(*_get_feed_data(args, input_data, user_triple_set, user_triple_set_e, item_triple_set, item_triple_set_i, 0, args.batch_size)).cpu().detach().numpy()
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
  
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
        
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))

            if (hit_num != 0):
                success_list[k].append(1)
            else:
                success_list[k].append(0)

            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(set(test_record[user])))

    model.train()

    hr = [np.mean(success_list[k]) for k in k_list]
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    _show_hr_info(zip(k_list, hr),f)
    _show_precision_info(zip(k_list, precision),f)
    _show_recall_info(zip(k_list, recall),f)

    
def _init_model(args, data_info):
    n_entity = data_info[2]
    n_relation = data_info[3]
    n_weight = args.buget_num + 2

    model = MEGA(args, n_entity, n_relation,n_weight)

    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    loss_func = nn.BCELoss()
    return model, optimizer, loss_func
    
    
def _get_feed_data(args, data, user_triple_set, user_triple_set_e,item_triple_set,item_triple_set_i, start, end):
    # origin item
    items = torch.LongTensor(data[start:end, 1])
    if args.use_cuda:
        items = items.cuda()
    # kg propagation embeddings
    user_triple = _get_triple_tensor(args, data[start:end,0], user_triple_set)
    user_triple_e = _get_triple_tensor(args, data[start:end,0], user_triple_set_e)
    items_triple = _get_triple_tensor(args, data[start:end,1], item_triple_set)
    items_triple_i = _get_triple_tensor(args, data[start:end,1], item_triple_set_i)

    return items,user_triple,user_triple_e, items_triple,items_triple_i


def _get_feed_label(args,labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.cuda()
    return labels


def _get_triple_tensor(args, objs, triple_set):
    # [h,r,t]  h: [layers, batch_size, triple_set_size]
    h,r,t = [], [], []
    for i in range(args.n_layer):
        h.append(torch.LongTensor([triple_set[obj][i][0] for obj in objs]))
        r.append(torch.LongTensor([triple_set[obj][i][1] for obj in objs]))
        t.append(torch.LongTensor([triple_set[obj][i][2] for obj in objs]))
        if args.use_cuda:
            h = list(map(lambda x: x.cuda(), h))
            r = list(map(lambda x: x.cuda(), r))
            t = list(map(lambda x: x.cuda(), t))
    return [h,r,t]


def _get_user_record(args, data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user,item])
    return np.array(res)


def _show_hr_info(hr_zip,f):
    res = ""
    for i,j in hr_zip:
        res += " K@%d:%.4f  "%(i,j)
    logging.info(res)
    f.write(res)
def _show_precision_info(pre_zip,f):
    res = ""
    for i,j in pre_zip:
        res += " K@%d:%.4f  "%(i,j)
    logging.info(res)
    f.write(res)
def _show_recall_info(recall_zip,f):
    res = ""
    for i,j in recall_zip:
        res += " K@%d:%.4f  "%(i,j)
    logging.info(res)
    f.write(res)


