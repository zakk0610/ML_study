import csv
import os
from shutil import copyfile
import numpy
from PIL import Image

data_path = os.getcwd()
images_path = data_path + '/picture/train'
data_path = data_path + '/resize_data'
if not os.path.isdir(data_path):
    os.mkdir('resize_data')

with open ('driver_imgs_list.csv') as csvfile:
    old_subject = ''
    old_class = ''
    index = -1
    sub_index = 0
    csvfile.readline()
    for row in csv.reader(csvfile, delimiter=','):
        #print 'old_subject=|'+old_subject+'|'
        #print 'old_class=|' + old_class +'|'
        #print 'input=|' +row[0] +'|'+row[1]+'|'+row[2] + '|'
        if row[0] != old_subject:
            index+=1
            old_subject = row[0]
            path = data_path+'/'+str(index) 
            if not os.path.isdir(path):
                os.mkdir(path)
            #print 'create '+ path
        if row[1] != old_class:
            old_class = row[1]
            path = data_path+'/'+str(index) +'/'+str(old_class) 
            if not os.path.isdir(path):
                os.mkdir(path)
            #print 'create ' + path
            sub_index=0
        src = images_path + '/' + row[1] + '/' + row[2]
        #print 'src=' + src
        dst = data_path+'/'+str(index)+'/'+str(old_class)+'/'+str(sub_index)+'.jpg'
        #print 'dst='+dst
#       img = Image.open(src).convert("L")
        img = Image.open(src)
        img = img.resize((224,224), Image.ANTIALIAS)
        img.save(dst)
        sub_index+=1


