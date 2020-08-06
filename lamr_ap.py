import os
import numpy as np
import brambox.boxes as bbb
import matplotlib.pyplot as plt
import scipy.interpolate

def meanAP_LogAverageMissRate():

    identify = lambda f: os.path.splitext("/".join(f.rsplit('/')[-3:]))[0]

    # parse ground truth from all videos in all sets
    ground_truth = bbb.parse('anno_dollar', 'annotations/*/*/*.txt', identify, occlusion_tag_map=[0.0, 0.25, 0.75])
    # print(len(ground_truth))
    # print(identify)
    # filter ground truth by marking boxes with the ignore flag
    bbb.filter_ignore(ground_truth, [bbb.ClassLabelFilter(['person']),  # only consider 'person' objects
                                     bbb.HeightRangeFilter((50, float('Inf'))),  # select instances of 50 pixels or higher
                                     bbb.OcclusionAreaFilter(
                                         (0.65, float('Inf')))])  # only include objects that are 65% visible or more

    for _, annos in ground_truth.items():
        for i in range(len(annos)):
            annos[i].class_label = 'person'

    # modify ground truth aspect ratio
    bbb.modify(ground_truth, [bbb.AspectRatioModifier(.41, modify_ignores=False)]);

    # split and copy to day and night ground truth
    ground_truth_day = {key: values for key, values in ground_truth.items() if
                        key.startswith('set06') or key.startswith('set07') or key.startswith('set08')}
    ground_truth_night = {key: values for key, values in ground_truth.items() if
                          key.startswith('set09') or key.startswith('set10') or key.startswith('set11')}


    def parse_detections(format, input, identify_fun=identify, clslabelmap=['person']):
        dets = bbb.parse(format, input, identify_fun, class_label_map=clslabelmap)
        bbb.modify(dets, [bbb.AspectRatioModifier(.41)])
        bbb.filter_discard(dets, [bbb.HeightRangeFilter((50 / 1.25, float('Inf')))])
        return dets


    detections_all = {}

    # detections_all['current'] = parse_detections('det_coco', 'results/conditioning/condition86e_mAP.json')
    # path_source = os.getcwd()
    # path_source = os.path.join(path_source, 'detection_results.json')
    path_source = 'results/detection_results.json'
    # print(path_source)
    detections_all['current'] = parse_detections('det_coco', path_source)

    # detections_all['Our: TD(V,V)'] = parse_detections('det_coco','results/adaptation/1_Visible_15e.json')
    # detections_all['Our: TD(T,T)'] = parse_detections('det_coco','results/adaptation/1_Thermal_15e.json')
    # detections_all['Our: TD(VT,T)'] = parse_detections('det_coco','results/adaptation/1_15e_toFT_34elike30e.json')
    # detections_all['Our: BU(VAT,T)'] = parse_detections('det_coco','results/adaptation/1_Adap30layers_From15_000014_best.json')
    # detections_all['Our: BU(VLT,T)'] = parse_detections('det_coco','results/adaptation/1_Layerwise5layers_from15e_000020_best.json')

    # detections_all['MSDS'] = parse_detections('det_coco','results/SOTA/MSDS.json')
    # detections_all['MSDS_sanitized'] = parse_detections('det_coco','results/SOTA/MSDS_sanitized.json')

    # split and copy to day and night detections
    detections_day = {}
    detections_night = {}
    for label, detections in detections_all.items():
        detections_day[label] = {key: values for key, values in detections.items() if
                                 key.startswith('set06') or key.startswith('set07') or key.startswith('set08')}
        detections_night[label] = {key: values for key, values in detections.items() if
                                   key.startswith('set09') or key.startswith('set10') or key.startswith('set11')}

    detectors_to_plot = ['current']
    # detectors_to_plot = ['Our: BU(VLT,T)', 'condition 86e map','Our: TD(V,V)','Our: TD(T,T)','Our: TD(VT,T)','Our: BU(VAT,T)','MSDS']

    def lamr(miss_rate, fppi, num_of_samples=9):
        """ Compute the log average miss-rate from a given MR-FPPI curve.
        The log average miss-rate is defined as the average of a number of evenly spaced log miss-rate samples
        on the :math:`{log}(FPPI)` axis within the range :math:`[10^{-2}, 10^{0}]`

        Args:
            miss_rate (list): miss-rate values
            fppi (list): FPPI values
            num_of_samples (int, optional): Number of samples to take from the curve to measure the average precision; Default **9**

        Returns:
            Number: log average miss-rate
        """
        samples = np.logspace(-2., 0., num_of_samples)
        m = np.array(miss_rate)
        f = np.array(fppi)
        interpolated = scipy.interpolate.interp1d(f, m, fill_value=(1., 0.), bounds_error=False)(samples)
        #     print('interpolated: ')
        #     print(interpolated)
        for i, value in enumerate(interpolated):
            if value <= 0:
                interpolated[i] = interpolated[i - 1]

        log_interpolated = np.log(interpolated)
        avg = sum(log_interpolated) / len(log_interpolated)
        return np.exp(avg)


    def generate_curves(ground_truth, results, pr=True, title="", saveplot="", overlap=0.5, only_plot=None,
                        linewidth=2, figsize=(8, 6), legendloc=3):
        curves = []
        scores = {}
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #     colors = ['#1919ff', '#ff7f0e', '#ff1919', '#ff19ff', '#19ff19', '#19ff19']
        colors = ['#1919ff', '#ff7f0e', '#ff1919', '#ff19ff', '#19ff19']
        i = 0
        linestyles = ['-', '--', '-.', ':']
        for label, detections in results.items():
            ### because YOLO has stuck in small object, so the paper of this code on CVPRW prefer chose overlap 0.4, but here we choose 0.5 for all.
            #         if label=='YOLO_TLV' or label=='Ours: TD(V,V)' or label=='Ours: TD(T,T)' or label=='Ours: TD(VT,T)' or label=='Ours: BU(VAT,T)' or label == 'Ours: BU(VLT,T)':
            #             print(label)
            #             overlap = 0.4
            if pr:
                ys, xs = bbb.pr(detections, ground_truth, overlap)
                score = round(bbb.ap(ys, xs) * 100, 2)
            else:
                ys, xs = bbb.mr_fppi(detections, ground_truth, overlap)
                score = round(lamr(ys, xs) * 100, 2)
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]

            if only_plot is None or label in only_plot:
                i += 1
            curves += [(label, ys, xs, score, color, linestyle)]
            scores[label] = score

        # if pr:
        #     # sort from highest ap to lowest
        #     sorted_curves = sorted(curves, key=lambda curve: curve[3], reverse=True)
        # else:
        #     # sort from lowest to highest
        #     sorted_curves = sorted(curves, key=lambda curve: curve[3])

        # fig, ax = plt.subplots(figsize=figsize)

        # for label, ys, xs, score, color, linestyle in sorted_curves:
        #     # skip curves not mensioned in only_plot
        #     if only_plot is not None and label not in only_plot:
        #         continue

        #     if pr:
        #         plt.plot(xs, ys, color=color, linestyle=linestyle, label=f"{score}%  {label}", linewidth=linewidth)
        #     else:
        #         plt.loglog(xs, ys, color=color, linestyle=linestyle, label=f"{score}%  {label}", linewidth=linewidth)
        #
        # plt.legend(loc=legendloc)
        #
        # plt.gcf().suptitle(title, weight='bold')

        # if pr:
        #     plt.grid(which='major')
        #     plt.gca().set_ylabel('Precision')
        #     plt.gca().set_xlabel('Recall')
        #     plt.gca().set_xlim([0, 1])
        #     plt.gca().set_ylim([0, 1])
        # else:
        #     # modify the y axis a bit
        #     from matplotlib.ticker import FormatStrFormatter, LogLocator
        #     subs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.4, 8.0]  # ticks to show per decade
        #     ax.yaxis.set_minor_locator(LogLocator(subs=subs))
        #     ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
        #     ax.yaxis.grid(which='minor')
        #     ax.xaxis.grid(which='major')
        #     plt.setp(ax.get_ymajorticklabels(), visible=False)  # disable major labels
        #
        #     plt.gca().set_ylabel('Miss rate')
        #     plt.gca().set_xlabel('FPPI')
        #     plt.gca().set_ylim([0.1, 1])
        #
            # plt.gca().set_xlim([0, 10])

        # if saveplot:
        # plt.savefig(saveplot, format='eps', dpi=1200)

        return scores


    scores_all_ap = generate_curves(ground_truth, detections_all, True, title="Day and night time",
                                    saveplot="all_pr.eps", only_plot=detectors_to_plot)
    scores_all_lamr = generate_curves(ground_truth, detections_all, False, title="Day and night time",
                                      saveplot="all_mr_fppi.eps", only_plot=detectors_to_plot)


    scores_day_ap = generate_curves(ground_truth_day, detections_day, True, title="Day time",
                                    saveplot="day_pr.eps", only_plot=detectors_to_plot, figsize=(8,6))
    scores_day_lamr = generate_curves(ground_truth_day, detections_day, False, title="Day time",
                                      saveplot="day_mr_fppi.eps", only_plot=detectors_to_plot, figsize=(8,6))


    scores_night_ap = generate_curves(ground_truth_night, detections_night, True, title="Night time",
                                      saveplot="night_pr.eps", only_plot=detectors_to_plot, figsize=(8,6))
    scores_night_lamr = generate_curves(ground_truth_night, detections_night, False, title="Night time",
                                        saveplot="night_mr_fppi.eps", only_plot=detectors_to_plot, figsize=(8,6), legendloc='lower left')
    # plt.show()

    # print(scores_all_ap['current'])
    # print(scores_day_ap['current'])
    # print(scores_night_ap['current'])
    # print(scores_all_lamr['current'])
    # print(scores_day_lamr['current'])
    # print(scores_night_lamr['current'])
    return  scores_all_ap['current'],scores_day_ap['current'],scores_night_ap['current'],scores_all_lamr['current'],scores_day_lamr['current'],scores_night_lamr['current']
