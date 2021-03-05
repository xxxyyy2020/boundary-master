#-*- coding: = utf-8 -*-
# author = lxj 
# date = 2020/4/27
"""
重新写一下，写成和英文的统一的pkl文件
将词典中的原因，跟在当前子句的下面
"""
import pickle
from xml.etree import ElementTree
import os
current_path = os.getcwd()
print('current_path = ',current_path)

def get_xml_data(para_xml_path, save_path):
    """
    :param para_xml_path:
    :param save_path:
    :return:
    """
    root = ElementTree.parse(para_xml_path)
    # 所有的节点列表
    lst_node = root.getiterator("emotionmll")
    # 对于每一个文档而言
    doc = lst_node[0]
    docNodes = doc.getchildren()
    # 文档的总个数
    num_doc = len(docNodes)
    print('num_doc = ', num_doc)
    # 记录所有的内容
    allList = []
    # 对每一个文档
    for index, child in enumerate(docNodes):
        data_doc = []
        currentCauseList = []
        currentClauseList = []
        currentKeyList = []

        docdic = dict()
        docID = index
        docdic['docID'] = docID
        data_doc.append(docdic)

        # for every clause
        for clause in child.getchildren():

            # 记录每一个文档里的ccategory和子句
            category = clause.getchildren()[0]
            categoryAttr = category.attrib
            # print('categoryAttr = ', categoryAttr)
            # 比当前文档的子句的个数多1, 对文档中的每一个子句；1是因为第一个孩子节点是category
            clauseNum = len(clause.getchildren())

            # 对文档中的每一个子句
            cause_index = 0
            for i in range(1, clauseNum):  # 对当前文档中的第i-1个子句
                clauseID = i
                clauseNode = clause.getchildren()[i]
                for cnode in clauseNode.getchildren():
                    attribu = cnode.attrib

                    # 存储当前的子句内容
                    if not attribu:
                        cause_content = ''
                        clauseContent = cnode.text.replace(' ', '').strip()
                        currentContentAttr = clauseNode.attrib
                        currentContentAttr['clauseID'] = clauseID
                        currentContentAttr['content'] = clauseContent
                        currentContentAttr['cause_content'] =cause_content

                        # 存储原因的属性和内容
                    if 'id' in attribu:
                        cause_index += 1

                        causeContent = cnode.text.replace(' ', '').strip()
                        currentCauseAttr = attribu
                        currentCauseAttr['index'] = cause_index
                        currentCauseAttr['cause_content'] = causeContent
                        currentCauseAttr['clauseID'] = clauseID
                        currentCauseList.append(currentCauseAttr)
                        currentContentAttr['cause_content'] = causeContent

                    # 存储关键词的属性和内容
                    if 'key-words-begin' in attribu:
                        # keywordsbegin = attribu['key - words - begin']
                        # keywordsLeng = attribu['keywords - length']
                        keyAttr = attribu
                        keywords = cnode.text.replace(' ', '').strip()
                        keyAttr['keyword'] =keywords
                        keyAttr['clauseID'] = clauseID
                        keyAttr['keyloc'] = clauseID - 1
                        # keyAttr['keyClause'] = clauseContent
                        currentKeyList.append(keyAttr)
                currentClauseList.append(currentContentAttr)

            clause_list = get_dis(currentKeyList[0], currentClauseList)
            data_doc.append(categoryAttr)
            data_doc.append(currentKeyList)
            data_doc.append(currentCauseList)
            data_doc.append(clause_list)

        allList.append(data_doc)
    print(allList[0])
    saveList(allList, save_path)
    return allList


# """
# [{'docId': 0}, {'name': 'surprise', 'value': '5'}, [{'keyword': 'was startled by', 'keyloc': 1, 'clauseID': 2}], [{'index': 1, 'cause_content': 'his unkempt hair and attire', 'clauseID': 2}], [{'cause': 'N', 'id': '1', 'keywords': 'N', 'clauseID': 1, 'content': 'That day Jobs walked into the lobby of the video game manufacturer Atari and told the personnel director'}, {'cause': 'Y', 'id': '2', 'keywords': 'Y', 'clauseID': 2, 'content': 'who was startled by his unkempt hair and attire'}, {'cause': 'N', 'id': '3', 'keywords': 'N', 'clauseID': 3, 'content': "that he wouldn ' t leave until they gave him a job"}]]
# [{'docID': 0}, {'name': 'happiness', 'value': '3'}, [{'key-words-begin': '0', 'keywords-length': '2', 'keyword': '激动', 'clauseID': 3, 'keyloc': 2}], [{'id': '1', 'type': 'v', 'begin': '43', 'length': '11', 'index': 1, 'cause_content': '接受并采纳过的我的建议', 'clauseID': 5}], [{'id': '1', 'cause': 'N', 'keywords': 'N', 'clauseID': 1, 'content': '河北省邢台钢铁有限公司的普通工人白金跃，', 'dis': -2}, {'id': '2', 'cause': 'N', 'keywords': 'N', 'clauseID': 2, 'content': '拿着历年来国家各部委反馈给他的感谢信，', 'dis': -1}, {'id': '3', 'cause': 'N', 'keywords': 'Y', 'clauseID': 3, 'content': '激动地对中新网记者说。', 'dis': 0}, {'id': '4', 'cause': 'N', 'keywords': 'N', 'clauseID': 4, 'content': '“27年来，', 'dis': 1}, {'id': '5', 'cause': 'Y', 'keywords': 'N', 'clauseID': 5, 'content': '国家公安部、国家工商总局、国家科学技术委员会科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议', 'dis': 2}]]
# """
def get_dis(para_key, para_clause):
    """NERDataset
    :param pra_key:
    :param para_clause:
    :return:
    """
    keyId = para_key['clauseID']
    returnlist = []
    for item in para_clause:
        aa = item
        clauseId = aa['clauseID']
        dis = clauseId - keyId
        item['dis'] = dis
        returnlist.append(item)
    return returnlist

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()

data_list = get_xml_data(para_xml_path =os.path.join(current_path, 'dataset/eca_ch/ECAEmnlp2016.xml'), save_path = os.path.join(current_path, 'dataset/eca_ch/eca_ch_data.pkl'))
print(data_list[0])






