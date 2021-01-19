import os
import argparse
import json
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pickle
FIGSIZE = (17, 11)
DPI = 300


def main():
    parser = argparse.ArgumentParser(description="plot_histogram")
    parser.add_argument("--data_root",
                        type=str,
                        required=True,
                        help="dataset root dir")
    parser.add_argument("--dataset_json",
                        type=str,
                        required=True,
                        help="dataset json file")
    parser.add_argument("--keys",
                        type=str,
                        required=True,
                        default="training,validation",
                        help="keys for dataset subset")
    parser.add_argument("--n_bins",
                        type=int,
                        required=False,
                        default=256,
                        help="number histogram bins")
    parser.add_argument("--range_min",
                        type=float,
                        required=False,
                        default=0.0,
                        help="minimum of histogram range")
    parser.add_argument("--range_max",
                        type=float,
                        required=False,
                        default=255.0,
                        help="maximum of histogram range")
    parser.add_argument("--output_root",
                        type=str,
                        required=False,
                        default="./histogram.png",
                        help="fileroot to save resulting histogram image and data")
    args = parser.parse_args()

    with open(args.dataset_json, 'r') as f:
        dataset = json.load(f)

    keys = args.keys.split(',')
    assert len(keys) > 0, "Specify at least one key!"

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    out_dict = {}
    for k, key in enumerate(keys):
        subset = dataset[key]
        histogram, bin_edges, n_images = compute_histo(args.data_root, subset, key, args.n_bins, args.range_min, args.range_max)

        # configure and draw the histogram figure
        plt.subplot(1, len(keys), k+1)
        plt.title(key)
        plt.xlabel("grayscale value")
        plt.ylabel("pixels")
        plt.xlim([args.range_min, args.range_max])
        #plt.plt(bin_edges[0:-1], histogram/np.max(histogram))
        plt.bar(bin_edges[0:-1], histogram/np.max(histogram), width=np.ptp(bin_edges)/args.n_bins, label=f'n={n_images}')

        #print(f"Histogram [{key}] for {n_images} images")
        #print(f'  bin_egdes: \n   {bin_edges}')
        #print(f'  histogram: \n   {histogram}')

        out_dict[key+'_histogram'] = histogram
        out_dict[key+'_bin_edges'] = bin_edges
        out_dict[key + '_n_images'] = n_images

        plt.legend()

    output_dir = os.path.dirname(args.output_root)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    #plt.show()
    plt.savefig(args.output_root+'.png')
    plt.savefig(args.output_root + '.svg')

    # save pickle
    with open(args.output_root+'.pkl', 'wb') as f:
        pickle.dump(out_dict, f)


def compute_histo(data_root, subset, key, n_bins, range_min, range_max):
    n_images = len(subset)
    assert n_images > 0, f"subset {key} is empty!"

    histogram = np.zeros((n_bins, ), dtype=np.int64)
    for i, datum in enumerate(subset):
        file = os.path.join(data_root, datum['image'])
        assert os.path.isfile(file), f"File {file} does not exist!"
        # print(f'adding {i + 1} of {len(subset)}: {file}')

        if '.npy' in file:
            I = np.load(file)
        elif '.png' in file:
            I = imageio.imread(file)
        else:
            raise ValueError(f'Image format not supported: {file}!')

        curr_histogram, bin_edges = np.histogram(I, bins=n_bins, range=(range_min, range_max))
        histogram += curr_histogram

    return histogram, bin_edges, n_images


if __name__ == "__main__":
    # execute only if run as a script
    main()
