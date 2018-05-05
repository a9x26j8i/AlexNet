# -*- coding:utf-8 -*-
import os
import random
import shutil
'''#随机抽取多个元素
import random
y=list(range(1,10))
slice = random.sample(y, 5)  #从list中随机获取5个元素，作为一个片断返回
print (slice)
print (y) #原有序列并没有改变。  
'''



def random_to_train_test(root_dir, name, dir, ratio):
    '''
    参数：
    name:  文件夹名字
    dir:   文件夹路径
    ratio: 测试集比率

    return:
    '''
    raw_list = os.listdir(dir)            #文件夹下文件list
    total_num = len(raw_list)             #文件夹下文件数量
    test_num = int(total_num * ratio)     #文件夹下作为测试集数量
    test_list = random.sample(raw_list,  test_num)

    if os.path.exists(root_dir + '/' + name + '_test'):
        print("%s test_path is exsists"%(name))
    else:
        os.mkdir(root_dir + '/' + name + '_test')

    for file in test_list:
        raw_dir = dir + '/' + file
        new_dir = root_dir + '/' + name + '_test' + '/' + file
        shutil.move(raw_dir, new_dir)

    print("%s test is done"%(name))


ROOT_DIR = '/home/xinlong/Tensorflow_workspace/canjian_AlexNet/JPG'
RATIO = 0.2

father_doc_list = os.listdir(ROOT_DIR)
for doc in father_doc_list:
    dir = ROOT_DIR + '/' + doc + '/'
    random_to_train_test(ROOT_DIR, doc, dir, RATIO)



