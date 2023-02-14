import argparse
import numpy as np
import logging
from collections import defaultdict
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

RATING_FILE_NAME = dict({'SH_S': 'method_api.dat', 'SH_L': 'method_api.dat', 'MV_S': 'method_api.dat'})
SEP = dict({'SH_S': '\t', 'SH_L': '\t', 'MV_S': '\t'})



def convert_rating(dataset):
    file = '../data/' + dataset + '/' + RATING_FILE_NAME[dataset]
    logging.info("reading rating file: %s", file)
    
    item_set = set()
    user_pos_ratings = defaultdict(list)
    
    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split(SEP[dataset])
        
        item_index = array[1]
        user_index = array[0]

        user_pos_ratings[user_index].append(item_index)
        item_set.add(item_index)

    write_file = '../data/' + dataset + '/ratings_final.txt'
    logging.info("converting rating file to: %s", write_file)
    writer = open(write_file, 'w', encoding='utf-8')
    writer_idx = 0
    user_cnt = 0
    for user_index, pos_item_set in user_pos_ratings.items():
        user_cnt += 1
        for item in pos_item_set:
            writer_idx += 1
            writer.write('%d\t%d\t1\n' % (int(user_index), int(item)))
        unwatched_set = item_set - set(pos_item_set)
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer_idx += 1
            writer.write('%d\t%d\t0\n' % (int(user_index), int(item)))
    writer.close()
    
    logging.info("number of users: %d", user_cnt)
    logging.info("number of items: %d", len(item_set))
    logging.info("number of interactions: %d", writer_idx)


if __name__ == '__main__':
    # we use the same random seed as RippleNet, KGCN, KGNN-LS for better comparison
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MV', help='which dataset to preprocess')
    args = parser.parse_args()

    convert_rating(args.dataset)

    logging.info("data %s preprocess: done.",args.dataset)

