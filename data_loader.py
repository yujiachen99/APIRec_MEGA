import collections
import os
import numpy as np
import logging
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== preparing data ===================")
    train_data, test_data, user_init_entity_set, item_init_entity_set,item_neighbor_item_dict = load_rating(args)
    n_user_entity, n_user_relation, user_kg, = load_user_kg(args)
    n_item_entity, n_item_relation, item_kg = load_item_kg(args)
    n_aitem_entity, n_aitem_relation, i2i_kg = load_i2i_kg(args)

    logging.info("contructing users' kg triple sets ...")
    user_triple_sets_i = kg_propagation(args, i2i_kg, user_init_entity_set, args.user_triple_set_size, True)
    user_triple_sets = kg_propagation(args, item_kg, user_init_entity_set, args.user_triple_set_size, True)
    logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, user_kg, item_init_entity_set, args.item_triple_set_size, False)
    item_triple_sets_i = kg_propagation(args, i2i_kg, item_neighbor_item_dict, args.item_triple_set_size, False)

    return train_data,  test_data, n_user_entity + n_item_entity, n_user_relation + n_item_relation, user_triple_sets, item_triple_sets, user_triple_sets_i,item_triple_sets_i


def build_item_to_item_graph(args,rating_np, train_indices,test_indices):

    if os.path.exists('../data/' + args.dataset + '/item_item_' + str(args.buget_num) + '.txt'):
        return 
    write_file = '../data/' + args.dataset + '/item_item_' +str(args.buget_num)+'.txt'
    logging.info("converting item_item file to: %s", write_file)
    user_history = defaultdict(list)
    item_set = set()
    # 先找出用户的历史记录
    for rating in rating_np:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if label == 1:
            user_history[user].append(item)
            item_set.add(item)

    relation = []
    nb_item = len(item_set)

    adj1 = [dict() for _ in range(nb_item+min(item_set))]
    adj = [[] for _ in range(nb_item+min(item_set))]
        
    indices = defaultdict(list)

    for user,item in user_history.items():
        if user in train_indices:
            indices[user] = item
        if user in test_indices:
            indices[user] = item[:4]

    seq = [val for _ , val in indices.items()]

    for i in range(len(seq)):
        data = seq[i]
        for k in range(1, 4):
            for j in range(len(data)-k):
                relation.append([data[j], data[j+k]])
                relation.append([data[j+k], data[j]])
    for tup in relation:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1
    

    weight = [[] for _ in range(nb_item+min(item_set))]

    for t in item_set:
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x] 
        weight[t] = [v[1] for v in x]

    for i in item_set:
        adj[i] = adj[i]
        weight[i] = weight[i]

    i2i_dic = defaultdict(list)
    weight_dic = defaultdict(list)
    # import pdb
    # pdb.set_trace()
    # 统计fre的最大值/最小值，进行分桶
    fre = []
    for iid, val in enumerate(weight):
        if iid < min(item_set):continue
        else:
            for i in val:
                fre.append(i)
    fre_min = min(fre)
    fre_max = max(fre)
    weight_hash = defaultdict(int)
    weigth = int((fre_max - fre_min) / args.buget_num) 
    for w in fre:
        weight_hash[w] = int ((w - fre_min) / weigth)
    # import pdb
    # pdb.set_trace()
    for iid, val in enumerate(adj):
        if iid < min(item_set):continue
        else:
            for i in val:
                i2i_dic[iid].append(i)
    
    for i in item_set:
        if i not in i2i_dic.keys():
            i2i_dic[i].append(i)

    
    for iid1, val1 in enumerate(weight):
        if iid1 < min(item_set):continue
        else:
            for i1 in val1:
                weight_dic[iid1].append(weight_hash[i1])
    
    for k in item_set:
        if k not in weight_dic.keys():
            weight_dic[k].append(weight_hash[1])


    writer = open(write_file, 'w', encoding='utf-8')


    for iid, val in i2i_dic.items():
        for idxx,i in enumerate(val):
            re = weight_dic[iid][idxx]
            # import pdb
            # pdb.set_trace()
            writer.write('%d\t%d\t%d\n' % (iid,re, i))

    writer.close()


def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file+str(args.buget_num) + '.npy', rating_np)
    return dataset_split(args,rating_np)


def dataset_split(args,rating_np):

    n_ratings = rating_np.shape[0]
    all_users = set(rating_np[:, 0])
    all_items = set(rating_np[:, 1])

    test_indices = []
    file = '../data/' + args.dataset + '/' + 'test_method'+'.txt'
    for line in open(file, encoding='utf-8').readlines():
        mid = line.strip()
        test_indices.append(int(mid))

    train_indices = list(set(all_users) - set(test_indices))

    build_item_to_item_graph(args,rating_np,train_indices,test_indices)
 
    user_init_entity_set, item_init_entity_set ,item_init_entity_set_i= collaboration_propagation(rating_np, train_indices,test_indices)

    train_data = []
    test_data = []
    for rating in tqdm(rating_np):
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if user in train_indices:
            train_data.append((user,item,label))
        if user in test_indices and item in user_init_entity_set[user]:
            train_data.append((user,item,label))
        if user in test_indices and item not in user_init_entity_set[user]:
            test_data.append((user,item,label))
    
    return train_data, test_data, user_init_entity_set, item_init_entity_set,item_init_entity_set_i
    
    
def collaboration_propagation(rating_np, train_indices,test_indices):

    user_history = defaultdict(list)
    # 先找出用户的历史记录
    for rating in rating_np:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if label == 1:
            user_history[user].append(item)

    item_history = defaultdict(list)
    for rating in rating_np:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if label == 1:
            item_history[item].append(user)

    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = defaultdict(list)
    item_history_user_dict = defaultdict(list)
    item_neighbor_item_dict = defaultdict(list)

    for user,itmess in user_history.items():
        if user in train_indices:
            itme_t = list(itmess)
        if user in test_indices:
            itme_t = list(itmess)[:4]
        for item in itme_t:
            user_history_item_dict[user].append(item)
            item_history_user_dict[item].append(user)

    logging.info("contructing items' initial entity set ...")
    for item in item_history_user_dict.keys():
        item_nerghbor_item = []
        for user in item_history_user_dict[item]:
            item_nerghbor_item = np.concatenate((item_nerghbor_item, user_history_item_dict[user]))
        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_history_user_dict:
            item_history_user_dict[item] = list(item_history[item])
            item_neighbor_item_dict[item] = [item]

    return user_history_item_dict, item_history_user_dict,item_neighbor_item_dict

def load_user_kg(args):
    kg_file = '../data/' + args.dataset + '/user_kg'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg

def load_item_kg(args):
    kg_file = '../data/' + args.dataset + '/item_kg'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg

def load_i2i_kg(args):
    kg_file = '../data/' + args.dataset + '/item_item_' + str(args.buget_num)
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg

def construct_kg(kg_np):
    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_set, set_size, is_user):
    # triple_sets: [n_obj][n_layer](h,r,t)x[set_size] 
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]

            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
                    
            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace= (len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets
