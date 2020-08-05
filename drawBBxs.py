from utils import *
from darknet import Darknet
import cv2
import tqdm

'''
Author: Kieu My
This file draws bounding box on the folder of images.
Input:
    (1):    Directory of images (image and annotation the same name, ex: I0001.png, I0001.txt)
    (2):    cfgfile is configuration file of the model
    (3):    weight / model file for the cfgfile model (if you saved weight or saved model)
Output: A folder with the same name of input folder name and '_predicted' with all images are drawn bounding boxes
        (a):    bounding boxes with green color is False Negative (which is groud truth that detector can not detect)
        (b):    bbxs with red color is False Positive (which detector detected but not in the ground-truth.
        (c):    bbxs with blue color is True Positive which is matched between detection and ground-truth
Noted that, TP, FP, FN here only for reference for drawing bboxes, not using for evaluation the detector. 
Recommend reader should check careful about this calculation if you use for evaluation. 
'''

namesfile=None
nms_thresh = 0.4
conf_thresh = 0.5
iou_thresh = 0.5
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

def IoU_boxes(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-torch.tensor(w1)/2.0, boxes2[0]-torch.tensor(w2)/2.0)
        x2_max = torch.max(boxes1[0]+torch.tensor(w1)/2.0, boxes2[0]+torch.tensor(w2)/2.0)
        y1_min = torch.min(boxes1[1]-torch.tensor(h1)/2.0, boxes2[1]-torch.tensor(h2)/2.0)
        y2_max = torch.max(boxes1[1]+torch.tensor(h1)/2.0, boxes2[1]+torch.tensor(h2)/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea / uarea)


def detect_model(cfgfile, modelfile,dir):

    m = Darknet(cfgfile)

    check_model = modelfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(modelfile)
        # print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
    else:
        m.load_weights(modelfile)

    # m.print_network()
    use_cuda = True
    if use_cuda:
        m.cuda()

    m.eval()

    class_names = load_class_names(namesfile)
    newdir = dir.replace('/','_') + 'predicted'
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    start = time.time()
    TPs,FPs,FNs,GTs = 0, 0, 0, 0
    for count, imgfile in enumerate(tqdm.tqdm(os.listdir(dir))):
        img_id,ext = os.path.basename(imgfile).split('.')
        if ext == 'txt':
            continue

        imgfile = os.path.join(dir,imgfile)

        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        lablepath = imgfile.replace('.jpg', '.txt').replace('.png', '.txt')

        if os.path.getsize(lablepath):
            truths = np.loadtxt(lablepath)
            truths = truths.reshape(truths.size // 5, 5)  # to avoid single truth problem
        else:
            truths = np.array([])
        new_truths = []
        for i in range(truths.shape[0]):
            new_truths.append([truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
        new_truths = np.array(new_truths)
        GTs += len(new_truths)

        detect_boxes = do_detect(m, sized, conf_thresh, nms_thresh, use_cuda)

        # print('Ground-truth bbxs = ', len(new_truths))
        # print('Detect bbxs = ', len(detect_boxes))

        groundtruth = []
        FN,TP = 0, 0
        for box_i in new_truths:
            check_TP = False
            for box_j in detect_boxes:
                if IoU_boxes(box_i, box_j, x1y1x2y2=False) >= iou_thresh:
                    TP += 1
                    check_TP = True
                    break
            if not check_TP:
                # print(imgfile, ' ', IoU_boxes(box_i, box_j, x1y1x2y2=False))
                groundtruth.append(box_i)
                FN += 1

        false_positive = []
        FP, TP = 0, 0
        for box_i in detect_boxes:
            # print(box)
            check_TP = False
            for box_j in new_truths:
                if IoU_boxes(box_i, box_j, x1y1x2y2=False) >= iou_thresh:
                    TP += 1
                    check_TP = True
                    break
            if not check_TP:
                # print(imgfile,' ',IoU_boxes(box_i, box_j, x1y1x2y2=False))
                false_positive.append(box_i)
                FP += 1
        # print('True Positive = %d \t False Positive = %d \t False Negative = %d \n', TP,FP,FN)

        for box_i in groundtruth:
            # print(box)
            for box_j in false_positive:
                if IoU_boxes(box_i, box_j, x1y1x2y2=False) >= iou_thresh:
                    FN -= 1
                    groundtruth.remove(box_i)
                    break

        TPs += TP
        FPs += FP
        FNs += FN


        plot_boxes_cv2(img, detect_boxes, class_names=class_names,color=blue)
        plot_boxes_cv2(img, false_positive, class_names=class_names,color=red)
        plot_boxes_cv2(img, groundtruth, class_names=class_names, color=green)

        savename = (imgfile.split('/')[-1]).split('.')[0]
        savename = savename + '_predicted.png'
        savename = os.path.join(newdir,savename)
        cv2.imwrite(savename, img)
    print('Ground-truth: %d \t True Positive: %d \t False Positive: %d \t False Negative: %d' % (GTs,TPs,FPs,FNs))
    # {: < 10s}\t{: .3f}
    print('Precision = %.2f' % (TPs/(TPs+FNs)))    ### detection / ground truth
    print('Precision theory: %.2f ' %(TPs / GTs))

    print('Recall = %.2f ' %(TPs/(TPs+FPs)))       ### detection / detection + wrong detection
    print('Missrate: %.2f ' %(FNs/(TPs+FNs)))      ### miss detection / ground-truth  = (1-Precision)
    print('Missrate theory = %.2f ' % (FNs / GTs))
    print('FPPI: %.2f' % (FPs/len(dir)))




    finish = time.time() - start
    print('Predicted in %d minutes %d seconds with average %f seconds / image.' % (finish//60, finish%60, finish/len(dir)))


if __name__ == '__main__':
    globals()["namesfile"] = 'data/kaist_person.names'

    cfgfile = 'cfg/yolov3_kaist_tc_det.cfg'
    modelfile = 'weights/yolov3_kaist_tc_det_thermal.model'

    if len(sys.argv) == 2:
        folder = sys.argv[1]
        if os.path.isdir(folder):
            detect_model(cfgfile, modelfile,folder)
    else:
        print('Usage: ')
        print('  python detect_folder.py foldername')
