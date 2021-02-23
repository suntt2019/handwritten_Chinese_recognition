#reference: https://morvanzhou.github.io/tutorials/

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import random

import cv2,os,sys
from skimage import transform

print('\n\n========================')
print("starting", sys.argv[0],"...")
if len(sys.argv)!=4:
    print("invalid argument count, the arguments should be:")
    print("  run_times background_color(\"bl\"/\"w\") rounds")
    os._exit(-1)
big_round_number=sys.argv[1]
print("big_round_number:",big_round_number)
bgcolor=sys.argv[2]
print("backgroundcolor:",bgcolor)
k_steps=sys.argv[3]
print("rounds: "+k_steps+'k('+str(int(float(k_steps)*1000))+')')

sl=40 #side_lenth
df='data_'+bgcolor+'_'+str(sl) #data_folder

test_images_arrays=[]
test_labels_arrays=[]
test_arrays=[]
train_images_arrays=[]
train_labels_arrays=[]
train_arrays=[]

save_file=np.load('savefile_'+df+'.npy')
test_x=save_file[0]
test_y=save_file[1]
train_x=save_file[2]
train_y=save_file[3]
branch_cnt,_,_=train_x.shape

def random_again():
    global train_x,train_y
    train_arrays=[]
    train_images_arrays=[]
    train_labels_arrays=[]
    for train_branch_num in range(0,branch_cnt):
        for train_item_num in range(0,50):
            train_img_and_label=[]
            train_img_and_label.append(train_x[train_branch_num][train_item_num])
            train_img_and_label.append(train_y[train_branch_num][train_item_num])
            train_arrays.append(train_img_and_label)
    random.shuffle(train_arrays)
    train_x_list=[]
    train_y_list=[]
    train_num=0
    for train_one in train_arrays:
        train_images_arrays.append(train_one[0])
        train_labels_arrays.append(train_one[1])
        train_num=train_num+1
        if train_num%50==0:
            train_x_list.append(np.array(train_images_arrays))
            train_images_arrays=[]
            train_y_list.append(np.array(train_labels_arrays))
            train_labels_arrays=[]
    train_x=np.array(train_x_list)
    train_y=np.array(train_y_list)

print('input finished!')

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

def get_proc_by_name(pname):

    for proc in psutil.process_iter():
        try:
            if proc.name().lower() == pname.lower():
                return proc  # return if found one
        except psutil.AccessDenied:
            pass
        except psutil.NoSuchProcess:
            pass
    return None

print('========================\n\n')

tf_x = tf.placeholder(tf.float32, [None,sl*sl]) / 255.
image = tf.reshape(tf_x, [-1, sl, sl, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 100])            # input y

# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
#print(type(7*7*32))
flat = tf.reshape(pool2, [-1, sl*sl*2])          # -> (7*7*32, )
output = tf.layers.dense(flat, 100)              # output layer

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph
with tf.Session():
    t1 = time.clock()
    t2 = time.clock()
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; ##########################print('\nPlease install sklearn for layer visualization\n')
def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

time_s=[]
loss_s=[]
accuracy_s=[]

CPU_temp_s=[]
CPU_freq_s=[]
CPU_percent_s=[]
RAM_usetage_s=[]

proc=get_proc_by_name('python3')

train_ad=0

for step in range(1,int(float(k_steps)*1000)+1):
    b_x=train_x[step%(branch_cnt-1)]
    b_y=train_y[step%(branch_cnt-1)]
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    
    if step % 100 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        loss_s.append(loss_)
        accuracy_s.append(accuracy_*100)
        CPU_freq=psutil.cpu_freq()[0];CPU_freq_s.append(CPU_freq)
        CPU_percent=psutil.cpu_percent(0);CPU_percent_s.append(CPU_percent)
        RAM_usetage=proc.memory_info()[0]/1024/1024;RAM_usetage_s.append(RAM_usetage)
        CPU_temp=psutil.sensors_temperatures()['coretemp'][0][1]
        #CPU_temp=0;
        CPU_temp_s.append(CPU_temp)
        time1 = time.clock()-t1
        time_s.append(time1)
        t1 = time.clock()
        
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f%%' % (accuracy_*100),'| time: %.3f'%time1,end='')
        print(' | CPU tempture: %.1f' %CPU_temp,'| CPU frequency: %.1f' %CPU_freq, 'MHz | CPU usetage: %.2f' % CPU_percent,'%% | RAM usetage: %.2f' % RAM_usetage,'MB')
        
        random_again()
        
        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
            low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
            labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
        
        
        
    
    if step % 1000 == 0:
        time2 = time.clock()-t2
        print('\n[big_round_number:'+big_round_number+']time last 1000 rounds=%.3f'%time2,'\n')
        t2 = time.clock()

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:100]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:100], 1), 'real number')

plt.ioff()

plt.figure(1, figsize=(30,20))

plt.suptitle('Training Statistic Data (Side Lenth: '+str(sl)+' | Background Color: '+bgcolor+' | Steps: '+k_steps+'k) ')


plt.subplot(231);plt.grid(True) 
plt.plot(time_s,"s-b",label="time") 
plt.title('time')

plt.subplot(232)
plt.plot(loss_s,"s-",color='orange',label="loss") 
plt.grid(True) 
plt.title('loss')

plt.subplot(233)
plt.plot(accuracy_s,"s-g",label="accuracy") 
plt.grid(True) 
plt.title('accuracy (final: '+str(accuracy_*100)[:5]+'%)')

plt.subplot(245)
plt.plot(CPU_temp_s,"s-",color='cadetblue',label="CPU tempture") 
plt.grid(True) 
plt.title('CPU tempture(Centigrade)')

plt.subplot(246)
plt.plot(CPU_freq_s,"s-",color='cadetblue',label="CPU frequency") 
plt.grid(True) 
plt.title('CPU frequency(MHz)')

plt.subplot(247)
plt.plot(CPU_percent_s,"s-",color='cadetblue',label="CPU usetage") 
plt.grid(True) 
plt.title('CPU usetage(all)')

plt.subplot(248)
plt.plot(RAM_usetage_s,"s-",color='sienna',label="RAM usetage") 
plt.grid(True) 
plt.title('RAM usetage(this program,MB)')

plt.savefig('./result/'+df+'_'+big_round_number+'_'+str(int(float(k_steps)*1000))+'_'+str(int(accuracy_*1000000))+'.jpg')

plt.close(1)

plt.figure(1, figsize=(15,10))

plt.plot(accuracy_s,"s-g",label="accuracy") 
plt.grid(True) 
plt.title('accuracy')

plt.savefig('./result/'+df+'_'+big_round_number+'_'+str(int(float(k_steps)*1000))+'_'+str(int(accuracy_*1000000))+'_a.jpg')


plt.close(1)

plt.figure(1, figsize=(15,10))

plt.suptitle('Training Statistic Data (Side Lenth: '+str(sl)+' | Background Color: '+bgcolor+' | Steps: '+k_steps+'k) ')

plt.subplot(231);plt.grid(True) 
plt.plot(time_s,"s-b",label="time") 
plt.title('time')

plt.subplot(232)
plt.plot(loss_s,"s-",color='orange',label="loss") 
plt.grid(True) 
plt.title('loss')

plt.subplot(233)
plt.plot(accuracy_s,"s-g",label="accuracy") 
plt.grid(True) 
plt.title('accuracy (final: '+str(accuracy_*100)[:5]+'%)')

plt.subplot(245)
plt.plot(CPU_temp_s,"s-",color='cadetblue',label="CPU tempture") 
plt.grid(True) 
plt.title('CPU tempture(Centigrade)')

plt.subplot(246)
plt.plot(CPU_freq_s,"s-",color='cadetblue',label="CPU frequency") 
plt.grid(True) 
plt.title('CPU frequency(MHz)')

plt.subplot(247)
plt.plot(CPU_percent_s,"s-",color='cadetblue',label="CPU usetage") 
plt.grid(True) 
plt.title('CPU usetage(all)')

plt.subplot(248)
plt.plot(RAM_usetage_s,"s-",color='sienna',label="RAM usetage") 
plt.grid(True) 
plt.title('RAM usetage(this program,MB)')

plt.savefig('./result/'+df+'_'+big_round_number+'_'+str(int(float(k_steps)*1000))+'_'+str(int(accuracy_*1000000))+'_small.jpg')