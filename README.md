# mlfunc_project
final project of W4995 Machine Learning for functional genomics

# Basic usage
Following show the basic usage to reproduce all experimental results. More detail usage will come soon

First, download the data and put them into a `./data/` folder 
```
. download.sh
```
Then run `. samlple.sh` to get different percentage of labelled data in each dataset
```
. sample.sh
```

To run the first experiment, simply run
```
. run_ref.sh
```

To run the second experiment, run
```
. run_paired.sh
```

To get all tsne plots, run
```
python tsne_plot.py
```

To plot the testing accuracy figures, use `acc_plot.py`. Following is an example that plot the testing accuray versus epoch in 15% labelled human pancreas dataset, change the input parameter to control which experiment you want to plot and where you want to save the results.
```
python acc_plot.py --res ./results/baron_2016h_labelled_15/ --save_path ./
```
