# train baron dataset
python train_paired.py --seed 10 --dataset_train ./data_sampled/baron_2016h_labelled_5.h5ad --dataset_test ./data/xin_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/baron_2016h_labelled_15.h5ad --dataset_test ./data/xin_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/baron_2016h_labelled_20.h5ad --dataset_test ./data/xin_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/baron_2016h_labelled_99.h5ad --dataset_test ./data/xin_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/baron_2016h_labelled_5.h5ad --dataset_test ./data/xin_2016.h5ad --num_gene_selected 200 --layers 100 50 --num_epoch 50 --save_path ./results_200g_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/baron_2016h_labelled_15.h5ad --dataset_test ./data/xin_2016.h5ad --num_gene_selected 200 --layers 100 50 --num_epoch 50 --save_path ./results_200g_paired --batch_size 128 --use_metric

# train hrvatin dataset
python train_paired.py --seed 10 --dataset_train ./data_sampled/hrvatin_2018_labelled_5.h5ad --dataset_test ./data/chen_2017.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/hrvatin_2018_labelled_15.h5ad --dataset_test ./data/chen_2017.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/hrvatin_2018_labelled_20.h5ad --dataset_test ./data/chen_2017.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/hrvatin_2018_labelled_99.h5ad --dataset_test ./data/chen_2017.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/hrvatin_2018_labelled_5.h5ad --dataset_test ./data/chen_2017.h5ad --num_gene_selected 200 --layers 100 50 --num_epoch 50 --save_path ./results_200g_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/hrvatin_2018_labelled_15.h5ad --dataset_test ./data/chen_2017.h5ad --num_gene_selected 200 --layers 100 50 --num_epoch 50 --save_path ./results_200g_paired --batch_size 128 --use_metric

# train macosko dataset
python train_paired.py --seed 10 --dataset_train ./data_sampled/macosko_2015_labelled_5.h5ad --dataset_test ./data/shekhar_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/macosko_2015_labelled_15.h5ad --dataset_test ./data/shekhar_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/macosko_2015_labelled_20.h5ad --dataset_test ./data/shekhar_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/macosko_2015_labelled_99.h5ad --dataset_test ./data/shekhar_2016.h5ad --num_gene_selected 2000 --layers 100 50 --num_epoch 50 --save_path ./results_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/macosko_2015_labelled_5.h5ad --dataset_test ./data/shekhar_2016.h5ad --num_gene_selected 200 --layers 100 50 --num_epoch 50 --save_path ./results_200g_paired --batch_size 128 --use_metric
python train_paired.py --seed 10 --dataset_train ./data_sampled/macosko_2015_labelled_15.h5ad --dataset_test ./data/shekhar_2016.h5ad --num_gene_selected 200 --layers 100 50 --num_epoch 50 --save_path ./results_200g_paired --batch_size 128 --use_metric
