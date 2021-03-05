#-*- coding: = utf-8 -*-
# author = lxj
#读取xml数据集
# date = 2020/3/2
import codecs
import regex as re
# path = 'EmotionCause.txt'
# inputFile = codecs.open(path, 'r','utf-8')
# aa = 0
# for item in inputFile:
#    aa = aa+1
#    # print(item)
#    a = r'<cause>(.*?)<\\cause>'
#    slotList = re.findall(a, item)
#    print(slotList)
#    # print(len(slotList))
#    if len(slotList) != 1:
#       print('hhhhh = ', len(slotList))
#
# print(aa)

def split_emo_data(data_path = 'EmotionCause.txt'):
   f = open(data_path)
   happtxt = []
   angertxt = []
   disgusttxt = []
   feartxt = []
   sadtxt = []
   shametxt =[]
   surprisetxt = []

   for line in f.readlines():
      if line.startswith('<happy>'):
         happtxt.append(line)

      elif line.startswith('<anger>'):
         angertxt.append(line)
      
      elif line.startswith('<disgust>'):
         disgusttxt.append(line)
      
      elif line.startswith('<fear>'):
         feartxt.append(line)
      
      elif line.startswith('<sad>'):
         sadtxt.append(line)
      
      elif line.startswith('<shame>'):
         shametxt.append(line)
      
      elif line.startswith('<surprise>'):
         surprisetxt.append(line)
   
   fh = open('happy.txt','w')
   fh.writelines(happtxt)
   fh.close()

   fa = open('anger.txt','w')
   fa.writelines(angertxt)
   fa.close()

   fd = open('disgust.txt','w')
   fd.writelines(disgusttxt)
   fd.close()

   ff = open('fear.txt','w')
   ff.writelines(feartxt)
   ff.close()

   fs = open('sad.txt','w')
   fs.writelines(sadtxt)
   fs.close()

   fsh = open('shame.txt','w')
   fsh.writelines(shametxt)
   fsh.close()

   fsu = open('surprise.txt','w')
   fsu.writelines(surprisetxt)
   fsu.close()

split_emo_data()

import pickle
def saveList(paraList, path):
   output = open(path, 'wb')
   # Pickle dictionary using protocol 0.
   pickle.dump(paraList, output)
   output.close()


def read_happy(data_path = 'happy.txt'):
   inputFile = codecs.open(data_path, 'r', 'utf-8')
   aa = 0
   happy_list, cause_list = [], []

   for item in inputFile:
      aa = aa + 1
      # print(item)
      item_text = item.replace('<cause>', '')
      item_text = item_text.replace('< cause >', '')
      item_text = item_text.replace('<\\cause>', '')
      a = r'<happy>(.*?)<\\happy>'
      slotList = re.findall(a, item_text)
      happy_list.append(slotList[0])

      item_cause = item.replace('<happy>', '')
      item_cause = item_cause.replace('< happy >', '')
      item_cause = item_cause.replace('<\\happy>', '')
      b = r'<cause>(.*?)<\\cause>'
      causeList = re.findall(b, item_cause)
      cause_list.append(causeList[0])

      if len(slotList) != 1:
         print('slotList = ', slotList )
         print('hhhhh = ', len(slotList))


   return happy_list, cause_list

happy_list, happy_cause_list = read_happy()
assert len(happy_list) == len(happy_cause_list)




def read_anger(data_path = 'anger.txt'):
   inputFile = codecs.open(data_path, 'r', 'utf-8')
   aa = 0
   happy_list, cause_list = [], []

   for item in inputFile:
      aa = aa + 1
      # print(item)
      item_text = item.replace('<cause>', '')
      item_text = item_text.replace('< cause >', '')
      item_text = item_text.replace('<\\cause>', '')
      a = r'<anger>(.*?)<\\anger>'
      slotList = re.findall(a, item_text)
      happy_list.append(slotList[0])

      item_cause = item.replace('<anger>', '')
      item_cause = item_cause.replace('< anger >', '')
      item_cause = item_cause.replace('<\\anger>', '')
      b = r'<cause>(.*?)<\\cause>'
      causeList = re.findall(b, item_cause)
      cause_list.append(causeList[0])

      if len(slotList) != 1:
         print('slotList = ', slotList )
         print('hhhhh = ', len(slotList))

   return happy_list, cause_list


anger_list, anger_cause_list = read_anger()

assert len(anger_list) == len(anger_cause_list)



def read_disgust(data_path = 'disgust.txt'):
   inputFile = codecs.open(data_path, 'r', 'utf-8')
   aa = 0
   happy_list, cause_list = [], []

   for item in inputFile:
      aa = aa + 1
      # print(item)
      item_text = item.replace('<cause>', '')
      item_text = item_text.replace('< cause >', '')
      item_text = item_text.replace('<\\cause>', '')
      a = r'<disgust>(.*?)<\\disgust>'
      slotList = re.findall(a, item_text)
      happy_list.append(slotList[0])

      item_cause = item.replace('<disgust>', '')
      item_cause = item_cause.replace('< disgust >', '')
      item_cause = item_cause.replace('<\\disgust>', '')
      b = r'<cause>(.*?)<\\cause>'
      causeList = re.findall(b, item_cause)
      cause_list.append(causeList[0])

      if len(slotList) != 1:
         print('slotList = ', slotList )
         print('hhhhh = ', len(slotList))

   return happy_list, cause_list


disgust_list, disgust_cause_list = read_disgust()

assert len(disgust_list) == len(disgust_cause_list)






def read_fear(data_path = 'fear.txt'):
   inputFile = codecs.open(data_path, 'r', 'utf-8')
   aa = 0
   happy_list, cause_list = [], []

   for item in inputFile:
      aa = aa + 1
      # print(item)
      item_text = item.replace('<cause>', '')
      item_text = item_text.replace('< cause >', '')
      item_text = item_text.replace('<\\cause>', '')
      a = r'<fear>(.*?)<\\fear>'
      slotList = re.findall(a, item_text)
      happy_list.append(slotList[0])

      item_cause = item.replace('<fear>', '')
      item_cause = item_cause.replace('< fear >', '')
      item_cause = item_cause.replace('<\\fear>', '')
      b = r'<cause>(.*?)<\\cause>'
      causeList = re.findall(b, item_cause)
      cause_list.append(causeList[0])

      if len(slotList) != 1:
         print('slotList = ', slotList )
         print('hhhhh = ', len(slotList))

   assert len(happy_list) ==len(cause_list)

   return happy_list, cause_list


fear_list, fear_cause_list = read_fear()

assert len(fear_list) == len(fear_cause_list)





def read_sad(data_path = 'sad.txt'):
   inputFile = codecs.open(data_path, 'r', 'utf-8')
   aa = 0
   happy_list, cause_list = [], []

   for item in inputFile:
      aa = aa + 1
      # print(item)
      item_text = item.replace('<cause>', '')
      item_text = item_text.replace('< cause >', '')
      item_text = item_text.replace('<\\cause>', '')
      a = r'<sad>(.*?)<\\sad>'
      slotList = re.findall(a, item_text)
      happy_list.append(slotList[0])

      item_cause = item.replace('<sad>', '')
      item_cause = item_cause.replace('< sad >', '')
      item_cause = item_cause.replace('<\\sad>', '')
      b = r'<cause>(.*?)<\\cause>'
      causeList = re.findall(b, item_cause)
      cause_list.append(causeList[0])

      if len(slotList) != 1:
         print('slotList = ', slotList )
         print('hhhhh = ', len(slotList))

   assert len(happy_list) ==len(cause_list)
   return happy_list, cause_list
sad_list, sad_cause_list =  read_sad()




def read_shame(data_path = 'shame.txt'):
   inputFile = codecs.open(data_path, 'r', 'utf-8')
   aa = 0
   happy_list, cause_list = [], []

   for item in inputFile:
      aa = aa + 1
      # print(item)
      item_text = item.replace('<cause>', '')
      item_text = item_text.replace('< cause >', '')
      item_text = item_text.replace('<\\cause>', '')
      a = r'<shame>(.*?)<\\shame>'
      slotList = re.findall(a, item_text)
      happy_list.append(slotList[0])

      item_cause = item.replace('<shame>', '')
      item_cause = item_cause.replace('< shame >', '')
      item_cause = item_cause.replace('<\\shame>', '')
      b = r'<cause>(.*?)<\\cause>'
      causeList = re.findall(b, item_cause)
      cause_list.append(causeList[0])

      if len(slotList) != 1:
         print('slotList = ', slotList )
         print('hhhhh = ', len(slotList))

   assert len(happy_list) ==len(cause_list)

   return happy_list, cause_list


shame_list, shame_cause_list =  read_shame()

def read_surprise(data_path = 'surprise.txt'):
   inputFile = codecs.open(data_path, 'r', 'utf-8')
   aa = 0
   happy_list, cause_list = [], []

   for item in inputFile:
      aa = aa + 1
      # print(item)
      item_text = item.replace('<cause>', '')
      item_text = item_text.replace('< cause >', '')
      item_text = item_text.replace('<\\cause>', '')
      a = r'<surprise>(.*?)<\\surprise>'
      slotList = re.findall(a, item_text)
      happy_list.append(slotList[0])

      item_cause = item.replace('<surprise>', '')
      item_cause = item_cause.replace('< surprise >', '')
      item_cause = item_cause.replace('<\\surprise>', '')
      b = r'<cause>(.*?)<\\cause>'
      causeList = re.findall(b, item_cause)
      cause_list.append(causeList[0])

      if len(slotList) != 1:
         print('slotList = ', slotList )
         print('hhhhh = ', len(slotList))

   assert len(happy_list) ==len(cause_list)
   return happy_list, cause_list

surprise_list, surprise_cause_list = read_surprise()


def writetidic(all_data, context_list, cause_list, strr):
   for text, cause in zip(context_list, cause_list):
      data_dict = dict()
      data_dict['id'] =  str(len(all_data))
      data_dict['emotion'] = strr
      data_dict['text'] = text
      data_dict['cause'] = cause
      all_data.append(data_dict)
   return all_data


all_data = []
all_data = writetidic(all_data, happy_list, happy_cause_list, strr = 'happy')
all_data = writetidic(all_data, anger_list, anger_cause_list, strr = 'anger')
all_data = writetidic(all_data, disgust_list, disgust_cause_list, strr = 'disgust')
all_data = writetidic(all_data, fear_list, fear_cause_list, strr = 'fear')
all_data = writetidic(all_data, sad_list, sad_cause_list, strr = 'sad')
all_data = writetidic(all_data, shame_list, shame_cause_list, strr = 'shame')
all_data = writetidic(all_data, surprise_list, surprise_cause_list, strr = 'surprise')

save_path = 'eca_sti.pkl'
saveList(all_data, save_path)
print(len(all_data))





