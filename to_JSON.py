'''
Author: Kieu My
'''
import os
import json
def convert_predict_to_JSON():
    # path_source = os.getcwd()
    path_source = 'results'

    filename = 'det_test_person.txt'

    desname = 'detection_results.json'
    if not os.path.exists(path_source):
        print('Error: No exits source folder')
        sys.exit()
    # os.chdir(path_source)

    f = open(os.path.join(path_source,filename),'r')
    lines = f.readlines()
    f.close()

    allscore = []
    alldata = []

    for line in lines:
        if len(line) > 1 or line != '\n':
            listdata = line.split(' ')
            imageID = listdata[0]
            confscore = float(listdata[1])
            left = float(listdata[2])
            top = float(listdata[3])
            right = float(listdata[4])
            bottom = float(listdata[5])
            allscore.append(confscore)

            ###this is only for KAIST dataset.
            imageID = imageID.replace('V','/V').replace('visible','/').replace('lwir','/').replace('_','')

            alldata.append({
                'image_id':imageID,
                'category_id':1,
                'bbox': [left,top,right-left,bottom-top],
                'score':confscore,
            })
    with open(os.path.join(path_source,desname), 'w') as outfile:
        json.dump(alldata, outfile,ensure_ascii=True)

    # minscore = min(allscore)
    # maxscore = max(allscore)
    # print('Max: {}'.format(maxscore))
    # print('Min: {}'.format(minscore))
    #
    # print("Conversion completed!")




if __name__ == '__main__':
    import sys
    if len(sys.argv) >=1:
        convert_predict_to_JSON()
    else:
        print('Usage:')
        print(' python convert_predict_YOLO_JSON.py')