from fishy_imports import *
from fishy_box import *

import math
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import time 

# training image path
t_image = list()

# validation image path
v_image = list()

# training label one hot encoded
train_label = list()

# validation label one hot encoded
valid_label = list()

for i in range(0,len(flist)):
    append_image_paths(i,flist[i],t_image,v_image)
    append_one_hot(i,flist[i],train_label,valid_label)


empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

for f in t_image:
    
    index = f.rfind('/')
    fname = f[index+1:]   
    
    if not fname in bb_json.keys(): 
        bb_json[fname] = empty_bbox

for f in v_image:
    
    index = f.rfind('/')
    fname = f[index+1:]   
    
    if not fname in bb_json.keys(): 
        bb_json[fname] = empty_bbox

print('finished appending empty boxes')


# generate pickle of rotated images with bounding boxes limit to 1000 images * 4 orientation per file 
# need to manually change idexes

print('train images',len(t_image))
print('validation images', len(v_image))

img_list = list()
bb_list = list()

#parameters to adjust 

start = 0
end = len(v_image)
blist_fname = 'valbblist.pkl'
ilist_fname = 'vallist.pkl'
      
for i in range(start,end):
    print(i)
    
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)
    
    img = Image.open(t_image[i])
    img1 = img.resize((299,299),Image.ANTIALIAS)
    img2 = img1.rotate(90, expand=True)
    img3 = img1.rotate(180, expand=True)
    img4 = img1.rotate(270, expand=True)
    
    img_list.append(img1)
    img_list.append(img2)
    img_list.append(img3)
    img_list.append(img4)
    
    size = img.size
    bb = show_bb(ax1,img1,t_image[i],bb_json,size)
    
    bb_list.append(bb)
    
    size = img1.size
    bb = rotate_bb(ax2,img2,bb,size,False)
    bb_list.append(bb)
    
    size = img2.size
    bb = rotate_bb(ax3,img3,bb,size,True)
    bb_list.append(bb)
    
    size = img3.size
    bb = rotate_bb(ax4,img4,bb,size,False)
    bb_list.append(bb)

#plt.show()

pickle.dump( img_list, open(ilist_fname,'wb'))
pickle.dump(  bb_list, open(blist_fname,'wb'))

print('combining complete')




