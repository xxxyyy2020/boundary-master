import pickle
import numpy as np
import os
current_path = os.getcwd()
print('current_path = ', current_path)

'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()

raw_data_path = os.path.join(current_path, 'dataset/eca_sti/eca_sti_data.pkl')
data_set = loadList(raw_data_path)
data_set_ids = list(range(len(data_set)))

np.random.seed()
np.random.shuffle(data_set_ids)

data_num = len(data_set_ids)
each_num = int(0.2 * len(data_set_ids))

dev_ids_each = []
for i in range(5):
    if i == 4:
        dev = data_set_ids[4*each_num: ]
    else:
        dev = data_set_ids[i * each_num: (i+1)*each_num]
    dev_ids_each.append(dev)

saveList(dev_ids_each, os.path.join(current_path, 'dataset/eca_sti/split_data_fold/eca_sti_dev_ids.pkl'))


