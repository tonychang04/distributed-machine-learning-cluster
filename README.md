# CS 425 Distributed Machine Learning Cluster(Best Rust MP * 4)
## Getting started
We are (proudly) using Rust! You can use [rustup](https://rustup.rs/) to setup your Rust environment easily.

## Usage
```shell
$ cargo run
```

Available CLI commands:
- `list_mem`, `lm`: list current membership list (only active members are shown)
- `list_self`: print current node's ID
- `join`, `j`: join the group
- `leave`, `l`: leave the group
- `p[ut] <local_file_path> <remote_filename>`: store a file in the file system
- `g[et] <remote_filename> <local_file_path>`: retrieve a file from the file system
- `d[elete] <remote_filename>`: delete a file from the file system
- `ls <remote_filename>`: list where a file is stored in the file system
- `s[tore]`: list all files stored in the current node
- `get-versions/gv <remote_filename> <num_versions> <local_file_path>`: get the last `num_versions` versions of a file
- `t[rain]`: train the machine learning models, in this case is loading the pretrained models 
- `predict`: perform distributed inference on imagenet_1k
- `jobs`: See the current status of the prediction jobs, including percentiles

The program logs to `HOSTNAME.log`.

### Report
Report is located [here](CS425MP4Report.pdf).

The data screenshots are located [here](data/patterns).


![13411683557337_ pic](https://user-images.githubusercontent.com/26497075/236856549-e13b036a-cfaf-462a-afcc-e14048485425.jpg)
