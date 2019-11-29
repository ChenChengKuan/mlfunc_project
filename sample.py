import numpy as np
import anndata
import os
import argparse
np.random.seed(10)

def sample(ref_file_name, output_path, sample_percentage):

    adata = anndata.read_h5ad(ref_file_name)
    cell_list = adata.obs['labels'].unique().tolist()
    adata_sampple_concat = []
    adata_unsample_concat = []
    print("Sample file : {}".format(ref_file_name))
    for cell in cell_list:
        adata_cell = adata[adata.obs['labels']== cell]
        shuffle_index = np.random.permutation(len(adata_cell))
        adata_cell_sampled = adata_cell[shuffle_index][0:int(len(adata_cell) * sample_percentage)]
        adata_cell_unsampled = adata_cell[shuffle_index][int(len(adata_cell) * sample_percentage):]
        adata_sampple_concat.append(adata_cell_sampled)
        adata_unsample_concat.append(adata_cell_unsampled)

    adata_labelled = anndata.AnnData.concatenate(*adata_sampple_concat)
    adata_unlabelled =  anndata.AnnData.concatenate(*adata_unsample_concat)

    ref_file_name_output = os.path.split(ref_file_name)[-1].split(".")[0]
    save_path_labelled = os.path.join(output_path, "{}_labelled_{}.h5ad".format(ref_file_name_output, sample_percentage))
    save_path_unlabelled = os.path.join(output_path, "{}_unlabelled_{}.h5ad".format(ref_file_name_output, sample_percentage))
    
    adata_labelled.write_h5ad(save_path_labelled)
    adata_unlabelled.write_h5ad(save_path_unlabelled)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_file_name', nargs='+', required=True)
    parser.add_argument('--output_path', default="./")
    parser.add_argument('--sample_rate', type=float, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)
    for ref_file in args.ref_file_name:
        sample(ref_file, args.output_path, args.sample_rate)

