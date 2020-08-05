from utils import *
from darknet import Darknet
import cv2

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    check_model = weightfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(weightfile)
        # print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
    else:
        m.load_weights(weightfile)

    namesfile = 'data/kaist_person.names'
    class_names = load_class_names(namesfile)
 
    use_cuda = True
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    if check_model == 'model':
        while True:
            res, img = cap.read()
            if res:
                sized = cv2.resize(img, (m.width, m.height))
                bboxes = do_detect_condition(m, sized, 0.5, 0.4, use_cuda)
                print('------')
                draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
                cv2.imshow(cfgfile, draw_img)
                cv2.waitKey(1)
            else:
                print("Unable to read image")
                exit(-1)

    else:
        while True:
            res, img = cap.read()
            if res:
                sized = cv2.resize(img, (m.width, m.height))
                bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                print('------')
                draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
                cv2.imshow(cfgfile, draw_img)
                cv2.waitKey(1)
            else:
                 print("Unable to read image")
                 exit(-1)

############################################
if __name__ == '__main__':
    cfgfile = 'cfg/yolov3_kaist.cfg'
    weightfile = 'weights/kaist_thermal_detector.weights'
    if len(sys.argv) >=1:
        if len(sys.argv) == 3:
            cfgfile = sys.argv[1]
            weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
    else:
        print('Usage:')
        print('    python demo.py [cfgfile] [weightfile]')
        print('    perform detection on camera')
