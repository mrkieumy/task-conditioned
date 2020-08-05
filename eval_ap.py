# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os,sys
# import cPickle
import _pickle as cPickle
# import pickle
import numpy as np
sys.path.append("stats")
from eval_all import get_det_result_name, get_image_txt_name
from utils import load_class_names


def convert_back_boundingbox(cx,cy,w,h):
    ''' This function to convert the bb as cx,cy,w,h of YOLO to original bb xmin,ymin,xmax,ymax'''
    img_w,img_h = 640,512
    cx *= img_w
    w *= img_w
    cy *= img_h
    h *= img_h
    xmin = cx - w/2.0
    ymin = cy - h/2.0
    xmax = cx + w/2.0
    ymax = cy + h/2.0
    return xmin,ymin,xmax,ymax

def parse_rec(filename,classlist):
    """ Parse a txt file from Faster RCNN of SOTA papers """
    with open(filename, 'r') as f:
        alllines = f.readlines()
    lines = [line.strip() for line in alllines]
    objects = []
    for line in lines:
        obj_struct = {}
        obj_struct['pose'] = '0'
        obj_struct['truncated'] = 0
        obj_struct['difficult'] = 0
        arr = line.split(' ')
        obj_id = int(arr[0])
        obj_struct['name'] = classlist[obj_id]
        cx = float(arr[1])
        cy = float(arr[2])
        w = float(arr[3])
        h = float(arr[4])
        xmin, ymin, xmax, ymax = convert_back_boundingbox(cx,cy,w,h)
        obj_struct['bbox'] = [xmin,ymin,xmax,ymax]
        objects.append(obj_struct)
    return objects


# def parse_rec(filename):
#     """ Parse a PASCAL VOC xml file """
#     tree = ET.parse(filename)
#     objects = []
#     for obj in tree.findall('object'):
#         obj_struct = {}
#         obj_struct['name'] = obj.find('name').text
#         obj_struct['pose'] = obj.find('pose').text
#         obj_struct['truncated'] = int(obj.find('truncated').text)
#         obj_struct['difficult'] = int(obj.find('difficult').text)
#         bbox = obj.find('bndbox')
#         obj_struct['bbox'] = [float(bbox.find('xmin').text),
#                               float(bbox.find('ymin').text),
#                               float(bbox.find('xmax').text),
#                               float(bbox.find('ymax').text)]
#         objects.append(obj_struct)
#
#     return objects

def eval_ap(rec, prec):
    """ ap = eval_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_recs_from_cache(imagenames, cachedir, cachename,classlist):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, cachename)

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(get_image_txt_name(imagename),classlist)
            #if i % 100 == 0:
            #    print ('Reading annotation for {:d}/{:d}'.format(
            #        i + 1, len(imagenames)))
        # save
        # print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        # print ('loaded cached annotations from {:s}'.format(cachefile))
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)
        try:
            for imagename in imagenames:
                recs[imagename]
        except Exception as e:
            print("Exception: {0}".format(e))
            print ('\t{:s} is corrupted. retry!!'.format(cachefile))
            os.remove(cachefile)
            recs = get_recs_from_cache(imagenames, cachedir, cachename,classlist)
    return recs

def get_class_det_result(detpath, classname):
    lines = []
    cls = classes.index(classname)
    imagename = None
    with open(detpath, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    lines = []
    for i, l in enumerate(splitlines):
            if l[0] == '#' and l[2] == '=': 
                if l[1] == 'imagepath':
                    imagename = l[3]
            elif l[0] != '' and l[0] != '#':
                if int(l[0]) == cls:
                    lines.append([imagename] +  l[1:])
    assert(imagename is not None)
    #print("{:s} {:s} {:d}".format(detpath, classname, len(lines)))
    return lines

def get_class_detection(imagenames, classname ):
    # load annots
    classlines = []
    for i, imagename in enumerate(imagenames):
        det = get_det_result_name(imagename)
        lines = get_class_det_result(det, classname)
        classlines.extend(lines)

    #print(classlines)
    ids = [x[0] for x in classlines]
    conf = np.array([float(x[1])for x in classlines])
    bb = np.array([[float(z)for z in x[2:]] for x in classlines])

    #print(ids)
    #print(bb)
    #print(conf)

    return ids, conf, bb

def eval(imagelist, classname, cachedir, classlist, ovthresh=0.5):
    """rec, prec, ap = eval(imagelist, classname, [ovthresh])
                                
    Top level function that does the PASCAL VOC evaluation.

    imagelist: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # read list of images
    with open(imagelist, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # cachedir caches the annotations in a pickle file
    recs = get_recs_from_cache(imagenames, cachedir, 'annots.pk',classlist)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    image_ids, confidence, BB = \
        get_class_detection(imagenames, classname )

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    #print(image_ids)
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        #print("%s (%s) " % (image_ids[d],classname), end='')
        #print(R)
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = eval_ap(rec, prec)

    return rec, prec, ap
    

def _do_python_eval(testlist, namelist, output_dir = 'output'):

    cachedir = os.path.join(output_dir, 'annotations_cache')
    aps = []
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    global classes
    classes = load_class_names(namelist)

    for i, cls in enumerate(classes):
        rec, prec, ap = eval(testlist, cls, cachedir, classes, ovthresh=0.5)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~~~~~~')
    print('   Results:')
    print('-------------')
    for i, ap in enumerate(aps):
        print('{:<10s}\t{:.3f}'.format(classes[i], ap))
    print('=============')
    print('{:^10s}\t{:.3f}'.format('Average', np.mean(aps)))
    print('~~~~~~~~~~~~~')
    print('')
    


if __name__ == '__main__':
    if len(sys.argv) == 3:
        testlist = sys.argv[1]
        namelist = sys.argv[2]
        _do_python_eval(testlist, namelist, output_dir = 'output')
    else:
        print("Usage: %s testlist namelist" % sys.argv[0] )
        

