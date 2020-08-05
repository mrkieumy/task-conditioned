'''
Authors: Kieu My
'''
from matplotlib import pyplot as pl

def readfilecondition(filename):
    with open(filename) as fp:
        data = fp.readlines()
    data = [item.rstrip() for item in data]
    tr_losses,classloss,classacc,fscores, precision,recall= [],[],[],[],[],[]
    for line in data:
        if line[0] != '#':
            try:
                epoch, _tr_loss, tr_loss, _closs, closs, _clacc, clacc, _score,score,_pre, pre, _rec, rec = line.split(' ')
                tr_losses.append(float(tr_loss))
                classloss.append(float(closs))
                classacc.append(float(clacc))
                fscores.append(float(score))
                precision.append(float(pre))
                recall.append(float(rec))
            except:
                print('expected: epoch: int yloss: float clsloss: float clsacc: float lr: float correct: float precision: float recall: float')
                print('received: ',line)

    return tr_losses,classloss,classacc,fscores,precision,recall

#### This function plot multi curves, each curve is one name
def plot_multicurves(list_curves, list_labels,start=0):
    list_colors = ['red', 'blue', 'green', 'black', 'pink', 'violet','yellow','cyan']
    ncurve = len(list_curves)
    min_len = min([len(list_curves[i]) for i in range(ncurve)])
    print('Plot ',ncurve,'curves from epoch ',start,' to epoch ',min_len)
    print('You can start from other epochs by add start value at the end of plot_multicurves(...,start) function')

    for i in range(ncurve):
        for j in range(start):
            list_curves[i][j] = list_curves[i][start]
        list_curves[i] = list_curves[i][:min_len]

    epoch = range(1,min_len+1)
    pl.figure(figsize=(8, 5))
    for i in range(ncurve):
        pl.plot(epoch, list_curves[i], color=list_colors[i], label=list_labels[i])
    pl.xlabel('number of epochs')
    pl.ylabel('percent')
    pl.grid(True)
    pl.legend()
    # pl.show()


def plot_one_curve(curve, name,start=0):
    print('Plot ', name, ' curve from epoch ', start, ' to the end ')
    print('You can start from other epochs by add start value at the end of plot_one_curve(...,start) function')
    print('epoch = ',len(curve))
    for i in range(start):
        curve[i] = curve[start]
    # curve = curve[start:]
    epoch = range(1,len(curve)+1)
    print('epoch = ',epoch)
    pl.figure(figsize=(8, 5))
    pl.plot(epoch, curve, color='red', label=name)
    pl.xlabel('epochs')
    pl.ylabel('value')
    pl.grid(True)
    pl.legend()
    # pl.show()

def scale0_1(curve):
    vmin = min(curve)
    vmax = max(curve)
    ncurve = [(x-vmin)/(vmax-vmin) for x in curve]
    return ncurve

def plot():
    ylosses, clslosses, clsacc, fscore, precision, recall = readfilecondition('backup/savelog.txt')
    plot_one_curve(ylosses,'training loss')
    ylosses = scale0_1(ylosses)
    plot_multicurves([ylosses, clslosses, clsacc], ['training loss', 'classify loss', 'classify accuracy'])

    plot_multicurves([fscore,precision, recall], ['validation fscore', 'validation precision','validation recall'])
    pl.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) >=1:
        plot()
    else:
        print('Usage:')
        print(' python seeloss_condition.py')
