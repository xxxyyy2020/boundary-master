import pickle
import string

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()

'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent


def remove_noise_strr_ch(content):
    """
    args:
        :param content:
        :return:
    """
    content = content.replace('  ', '')
    content = content.replace('@', '')
    content = content.replace('#', '')
    content = content.replace('<', '')
    content = content.replace('>', '')
    content = content.replace('=', '')
    content = content.replace('`', '')
    content = content.replace('\n', '')
    content = content.replace('\t', '')
    content = content.replace('*', '')
    content = content.replace('&', '')
    content = content.replace(' ', '')
    return content.strip()

def get_clean_data_ch(para_text):
    returntext = remove_noise_strr_ch(para_text)
    return returntext

def remove_noise_strr_en(content):
    """
    args:
        :param content:
        :return:
    """
    
    content = content.replace('  ', ' ')
    content = content.replace('@', '')
    content = content.replace('#', '')
    content = content.replace('<', '')
    content = content.replace('>', '')
    content = content.replace('=', '')
    content = content.replace('`', '')
    content = content.replace('\n', '')
    content = content.replace('\t', '')
    content = content.replace('*', '')
    content = content.replace('&', '')
    content = content.replace('  ', ' ')
    content = content.replace('  ', ' ').lower()

    # print('string.punctuation = ', string.punctuation)
    punct = string.punctuation + '“' + '‘'

    table = str.maketrans({key: ' ' + key + ' ' for key in punct})
    # print('table = ', table)
    content = content.translate(table)
    content = content.replace('  ', ' ').strip() #去掉多余空格
    content_l = content.split()
    cont = ''
    for item in content_l:
        if item != '':
            cont += item
            cont += ' '
    content = cont.replace('  ',' ')
    return content.strip()


def get_clean_data_en(para_text):
    returntext = remove_noise_strr_en(para_text)
    return returntext



