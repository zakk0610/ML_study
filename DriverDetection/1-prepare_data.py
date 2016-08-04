import csv
import os
from shutil import copyfile

data_path = os.getcwd()
src_path = data_path + '/resize_data'
training_data = data_path + '/training_set'
validation_data = data_path + '/validation_set'

if not os.path.isdir(training_data):
    os.mkdir(training_data)
    for i in range(0,10):
        os.mkdir(training_data +"/"+str(i))
if not os.path.isdir(validation_data):
    os.mkdir(validation_data)
    for i in range(0,10):
        os.mkdir(validation_data +"/"+str(i))

with open ('driver_imgs_list.csv') as csvfile:
    old_subject = ''
    old_class = ''
    index = -1
    sub_index = 0
    global_idx = 0
    csvfile.readline()
    for row in csv.reader(csvfile, delimiter=','):
        if row[0] != old_subject:
            index+=1
            old_subject = row[0]
        if row[1] != old_class:
            old_class = row[1]
            sub_index=0
        if index <= 24:
           src = src_path+'/'+str(index)+'/'+str(old_class)+'/'+str(sub_index)+'.jpg'
           print 'src='+src
           dst = training_data+'/'+str(old_class)[-1]+'/'+str(global_idx)+'.jpg'
           print 'dst='+dst
           copyfile(src,dst)
        else:   
           src = src_path+'/'+str(index)+'/'+str(old_class)+'/'+str(sub_index)+'.jpg'
           print 'src='+src
           dst = validation_data+'/'+str(old_class)[-1]+'/'+str(global_idx)+'.jpg'
           print 'dst='+dst
           copyfile(src,dst)
        sub_index+=1
        global_idx+=1


