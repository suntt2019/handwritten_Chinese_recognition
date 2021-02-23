import numpy as np
import time, psutil, random, sys, os, cv2

print("starting", sys.argv[0],"...")
if len(sys.argv)!=4:
    print("invalid argument count, the arguments should be:")
    print("  background_color(\"bl\"/\"w\")  side_lenth  group_count")
    os._exit(-1)
background_color=sys.argv[1]
print("background_color:",background_color)
sl=int(sys.argv[2])
print("side_lenth:",sl)
group_count=int(sys.argv[3])
print("group_count:",group_count)

#sl=100 #side_lenth
df='data_'+background_color+'_'+str(sl) #data_folder

test_images_arrays=[]
test_labels_arrays=[]
test_arrays=[]
train_images_arrays=[]
train_labels_arrays=[]
train_arrays=[]

def pic_into_array(file):
    array=[]
    img=cv2.imread(file,0)
    #print(file)
    for row in range(sl):
        for col in range(sl):
            array.append(np.float32(img[row][col]/255))
            nparray=np.array(array)
    return nparray

print('Loading test data...')
for filenum in range(0,group_count):
    print('Loading test-',filenum)
    os.chdir('./'+df+'/test/'+str(filenum).zfill(5))
    filenames = os.listdir()
    os.chdir('..');os.chdir('..');os.chdir('..')
    for i in range(len(filenames)):
        file='./'+df+'/test/'+str(filenum).zfill(5)+'/'+filenames[i]
        test_labels_array=[]
        for j in range(100):
            if filenum==j:
                test_labels_array.append(1)
            else:
                test_labels_array.append(0)
        test_img_and_label=[]
        test_img_and_label.append(pic_into_array(file))
        test_img_and_label.append(np.array(test_labels_array))
        test_arrays.append(test_img_and_label)

print('Loading train data...')
for filenum in range(0,group_count):
    print('Loading train-',filenum)
    os.chdir('./'+df+'/train/'+str(filenum).zfill(5))
    filenames = os.listdir()
    os.chdir('..');os.chdir('..');os.chdir('..')
    for i in range(len(filenames)):
        file='./'+df+'/train/'+str(filenum).zfill(5)+'/'+filenames[i]
        train_labels_array=[]
        for j in range(100):
            if filenum==j:
                train_labels_array.append(1)
            else:
                train_labels_array.append(0)
        train_img_and_label=[]
        train_img_and_label.append(pic_into_array(file))
        train_img_and_label.append(np.array(train_labels_array))
        train_arrays.append(train_img_and_label)

print('random...')
random.shuffle(test_arrays)
random.shuffle(train_arrays)

print('Packaging ')
for test_one in test_arrays:
    test_images_arrays.append(test_one[0])
    test_labels_arrays.append(test_one[1])
train_num=0
train_x_list=[]
train_y_list=[]
for train_one in train_arrays:
    train_images_arrays.append(train_one[0])
    train_labels_arrays.append(train_one[1])
    train_num=train_num+1
    if train_num%50==0:
        train_x_list.append(np.array(train_images_arrays))
        train_images_arrays=[]
        train_y_list.append(np.array(train_labels_arrays))
        train_labels_arrays=[]
test_x=np.array(test_images_arrays)[:2000]
test_y=np.array(test_labels_arrays)[:2000]
train_x=np.array(train_x_list)
train_y=np.array(train_y_list)

save_file=np.array([test_x,test_y,train_x,train_y])
np.save('savefile_'+df+'.npy',save_file)
