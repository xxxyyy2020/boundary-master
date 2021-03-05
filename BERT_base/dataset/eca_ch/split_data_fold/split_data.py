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

raw_data_path = os.path.join(current_path, 'dataset/eca_ch/eca_ch_data.pkl')
data_set = loadList(raw_data_path)

data_set_id = list(range(0, len(data_set))) #是一个list列表[0,1,2,3,4,5,...]
dev_l = []
for i in range(20):
    np.random.seed()
    np.random.shuffle(data_set_id)
    test_num = int(len(data_set_id) * 0.1)
    test_id = data_set_id[0:test_num]
    dev_l.append(test_id)

saveList(dev_l, os.path.join(current_path, 'dataset/eca_ch/split_data_fold/eca_ch_dev_id.pkl'))



