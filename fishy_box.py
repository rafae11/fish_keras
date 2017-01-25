from utils import *
import ujson as json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

path = '/media/rafael/Data1/sample/'
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']

bb_json = {}
for c in anno_classes:
    j = json.load(open('{}annos/{}_labels.json'.format(path, c), 'r'))
        
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]


empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = 299 / size[0]
    conv_y = 299 / size[1]
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    
    return bb

def create_rect(bb, color='red'):
    return patches.Rectangle((bb[2],bb[3]),bb[1],bb[0], color=color, fill=False, lw=3)

def show_bb(ax,img,path,bb_json,size):
    ax.imshow(img)
    index = path.rfind('/')
    fname = path[index+1:]
    bb = convert_bb(bb_json[fname],size)
    ax.add_patch(create_rect(bb))
    return bb

def rotate_bb(ax,img,bb,size,flip):
    ax.imshow(img)

    if(flip == True):

        newx = bb[3]
        newy = size[0] - bb[2] - bb[1]
        neww = bb[1]
        newh = bb[0]
    else:
        newx = bb[3]
        newy = size[1] - bb[2] - bb[1]
        neww = bb[1]
        newh = bb[0]

    bb[0] = neww
    bb[1] = newh
    bb[2] = newx
    bb[3] = newy

    ax.add_patch(create_rect(bb))
    return bb
