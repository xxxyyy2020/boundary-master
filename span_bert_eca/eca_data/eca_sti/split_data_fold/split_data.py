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

raw_data_path = os.path.join(current_path, 'eca_data/eca_sti/eca_sti_data.pkl')
data_set = loadList(raw_data_path)

np.random.seed()
np.random.shuffle(data_set)
# test_num = int(len(data_set) * 0.1)
# test_data = data_set[0 : test_num]
# saveList(test_data, os.path.join(current_path, 'dataset/eca_sti/split_data_fold/eca_sti_test_example.pkl'))

# other_data = data_set[test_num : ]
data_num = len(data_set)
each_num = int(0.2 * len(data_set))

for i in range(5):
    if i == 4:
        dev = data_set[4*each_num: ]
        saveList(dev, os.path.join(current_path, 'eca_data/eca_sti/split_data_fold/eca_sti_dev{}_example.pkl'.format(i)))
    else:
        # print('ssss')
        dev = data_set[i * each_num: (i+1)*each_num]
        # print('ddd')
        saveList(dev, os.path.join(current_path, 'eca_data/eca_sti/split_data_fold/eca_sti_dev{}_example.pkl'.format(i)))


        