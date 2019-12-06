import glob
import torch
from sklearn.manifold import TSNE
import glob
import json
import pandas as pd
import seaborn as sns
import argparse
import os
import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=str, required=True)
    args = parser.parse_args()
    embed_res = glob.glob(os.path.join("{}".format(args.res), "*", "*.pt")) 
    embed_meta_res = glob.glob(os.path.join("{}".format(args.res), "*", "*", "*","metadata.tsv"))

for embed, meta in zip(embed_res, embed_meta_res):
    print("plotting ", embed)
    embed_ckpt = torch.load(embed).numpy()
    cell_labels = []
    with open(meta) as handle:
        for line in handle.readlines():
            cell_labels.append(int(line.strip().split("-")[0]))
    
    with open(embed.replace("embed.pt", 'id2cell.json'),'r') as handle:
        id2cell = json.load(handle)
    cell_labels = [id2cell[str(lb)] for lb in cell_labels]
    tsne = TSNE(verbose=1)
    embed_transform = tsne.fit_transform(embed_ckpt)
    
    embed_df = pd.DataFrame(embed_transform)
    embed_df.columns = ['tsne_one', 'tsne_two']
    embed_df['cell_type'] = cell_labels
    plt.figure(figsize=(16,10))
    sns.scatterplot(x='tsne_one', y='tsne_two',\
                   data=embed_df, hue='cell_type',\
                   palette=sns.color_palette("hls", 9))
    plt.savefig(embed.replace('embed.pt', 'tsne_embed'))

