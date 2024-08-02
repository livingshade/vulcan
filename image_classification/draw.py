import matplotlib.pyplot as plt
import json
import sys
import os
import numpy as np
import time


color_map = {
    "random": "red",
    "stratified": "blue",
}

knobs = {
    'model': ['resnet34', 'resnet50', 'resnet101']
}

methods = ["random", "stratified"]


def draw_average(records, per, prefix=""):
    plt.cla()
    figures_dict = {}
    num_fig = 0
    for r in records:
        model = str(r['args'])
        method = str(r['method'])

        if model in knobs["model"]:
            key = f"{model}"
            if key not in figures_dict:
                num_fig += 1
                figures_dict[key] = []
                print(f"model: {model}, method: {method}")
            figures_dict[key].append(r)

    print("Num figure", num_fig)
    fig, axs = plt.subplots(num_fig, figsize=(15,15))
    fig.tight_layout(h_pad=3)    

    if num_fig == 1:
        axs = [axs]
        
    TOTAL = 50000
    INTERVAL = 100
    for axid, k in enumerate(figures_dict.keys()):
        ax = axs[axid]
        cul_acc1s = {
            "random": [[] for i in range(TOTAL // INTERVAL)],
            "stratified": [[] for i in range(TOTAL // INTERVAL)],
        }
        cul_acc5s = {
            "random": [[] for i in range(TOTAL // INTERVAL)],
            "stratified": [[] for i in range(TOTAL // INTERVAL)],
        }
        # ax = fig.add_subplot()
        for idx, r in enumerate(figures_dict[k]):
            record = r
            sample_idx = record['idx']
            method = record['method']
            model = record['args']
            cul_acc1 = record['cul_acc1']
            cul_acc5 = record['cul_acc5']


            xid = range(TOTAL // INTERVAL)
            
            for i in range(TOTAL):
                if i % INTERVAL == 0:
                    cul_acc1s[method][i // INTERVAL].append(cul_acc1[i])
                    cul_acc5s[method][i // INTERVAL].append(cul_acc5[i])
            # ax.plot(xid, cul_acc1[skip:], label=method, color=color_map[method], linewidth=0.1)
                # print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, method: {method}")
            if sample_idx == 0 and method == "random":
                ax.set_xlabel('# sample')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{model}')
                ax.set_xticks([i for i in xid if i % 10 == 0], [str(i * INTERVAL) for i in xid if i % 10 == 0])
                ax.set_xlim(0, len(xid) - 1)
                # plt.axhline(y=ra_acc, color='g', linestyle='dashed', label='acc')
                ground_truth = cul_acc1[-1]
                ax.axhline(y=ground_truth + 0.01, color='grey', linestyle='dashed', label='acc+1%')
                ax.axhline(y=ground_truth - 0.01, color='grey', linestyle='dashed', label='acc-1%')
                ax.set_ylim(ground_truth - 0.05, ground_truth + 0.05)
                
        def average(lst):
            return sum(lst) / len(lst)
        def variance(lst):
            return sum([(x - average(lst))**2 for x in lst]) / len(lst)
        def percentile(lst, p):
            return lst[int(float(len(lst)) * (p / 100.0))]
        
        for method in methods:
            upper_bound = []
            lower_bound = []
            for i in range(len(cul_acc1s[method])):
                acc = sorted(cul_acc1s[method][i])
                upper_bound.append(percentile(acc, per))
                lower_bound.append(percentile(acc, 100 - per))
            ax.plot(xid, lower_bound, color=color_map[method], label=method,linewidth=0.7)
            ax.plot(xid, upper_bound, color=color_map[method],linewidth=0.7)
            # if method == "random":
                # print(upper_bound[:3], lower_bound[:3])
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="right")
    plt.savefig(f"./result/figures/{prefix}.png")
    print(f"Save to ./result/figures/{prefix}.png")

    
if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    date_expect = sys.argv[1]
    assert(len(date_expect) > 0)


    records = []
    for (dirpath, dirname, filenames) in os.walk("./result/"):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            method = filename.split("_")[0]
            date = filename.split("_")[-1].split(".")[0]
            # if date != date_expect:
            #     continue
            print("Load file: ", filename)
            # idx = filename.split("_")[1]
            with open(f'./result/{filename}', 'r') as fp:
                record = json.load(fp)
                for r in record:
                    r["method"] = method
                    r["cul_acc1"] = [x / 100 for x in r["cul_acc1"]]
                records += record
                
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    # draw_new(records, f"all_{date_time}")

    draw_average(records, 80, f"avg_p95_{date_time}")
    # draw_average(records, 99, f"avg_p99_{date_time}")

    # plot_new(records_random[0], f"random_0")