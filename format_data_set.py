import cv2,os,sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import shutil

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
'''
sl=100 #side_lenth
background_color='bl'
'''
def normalize_pic(file):
	#print(file)
	img=cv2.imread(file,0)
	rows,cols=img.shape
	maxsize=max(rows,cols)
	rate=sl/maxsize
	rows1=(int)(rows*rate)
	cols1=(int)(cols*rate)
	mid=cv2.resize(img,(cols1,rows1),interpolation=cv2.INTER_CUBIC)
	#print('rows1',rows1,'cols1',cols1)
	left_add=(int)((sl-cols1)/2)
	top_add=(int)((sl-rows1)/2)
	right_add=sl-cols1-left_add
	bottom_add=sl-rows1-top_add
	#print('top',top_add,'bottom',bottom_add,'left',left_add,'right',right_add)
	ret=cv2.copyMakeBorder(mid,top_add,bottom_add,left_add,right_add,cv2.BORDER_CONSTANT,value=255)
	if background_color == 'bl':
		for row in range(sl):
			for col in range(sl):
				ret[row][col]=255-ret[row][col]
		#plt.imshow(ret)
	#plt.show()
	return ret

data_dir='data_'+background_color+'_'+str(sl)

if os.path.exists(data_dir):
	print("output directory exists, type [Y/y] to remove")
	inp=input()
	if inp=="y" or inp=="Y":
		shutil.rmtree(data_dir)
	else:
		os._exit(0)


os.mkdir(data_dir)
os.mkdir(data_dir+'/test')

for filenum in range(0,group_count):
	os.chdir('./data/test/'+str(filenum).zfill(5))
	filenames = os.listdir()
	os.chdir('..');os.chdir('..');os.chdir('..')
	os.mkdir(data_dir+'/test/'+str(filenum).zfill(5))
	for i in range(len(filenames)):
		file='test/'+str(filenum).zfill(5)+'/'+filenames[i]
		ret=normalize_pic('./data/'+file)
		cv2.imwrite('./'+data_dir+'/'+file,ret)
	print('finished',filenum,file)

os.mkdir(data_dir+'/train')

for filenum in range(0,group_count):
	os.chdir('./data/train/'+str(filenum).zfill(5))
	filenames = os.listdir()
	os.chdir('..');os.chdir('..');os.chdir('..')
	os.mkdir(data_dir+'/train/'+str(filenum).zfill(5))
	for i in range(len(filenames)):
		file='train/'+str(filenum).zfill(5)+'/'+filenames[i]
		ret=normalize_pic('./data/'+file)
		cv2.imwrite('./'+data_dir+'/'+file,ret)
	print('finished',filenum,file)


