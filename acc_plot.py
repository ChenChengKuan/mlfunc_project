import json
import matplotlib.pyplot as plt
import os
import glob
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=str, nargs='+', required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--paired', default=False, action='store_true')
    args = parser.parse_args()
    if not args.paired:
        log_dir_srt = sorted(args.res, key = lambda x: -int(x.split("/")[2].split("_")[-1]))
    else:
        log_dir_srt = sorted(args.res, key=lambda x: -int(x.split("/")[2].split("_")[-3]))
    for file in log_dir_srt:
        tmp = glob.glob(os.path.join(file, '*', 'log.json'))[0]
        with open(tmp)as f:
            legend_flag = file.split("/")[2].split("_")[-3]
            results = json.load(f)
            log = results['log']
            test_acc = []
            for i in range(1,len(log),2):
                test_acc.append(log[i]['test_acc'])
            plt.plot(list(range(len(test_acc))), test_acc, label='{}_labelled'.format(legend_flag))
    plt.legend()
    plt.xlabel("Number of epoch")
    plt.ylabel("Test accuracy")
    data_name = "_".join(file.split("/")[2].split("_")[0:-2])
    plt.savefig(os.path.join(args.save_path, "{}_test_acc_plot".format(data_name)))