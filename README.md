# FocusBranch

Official code repo for **FocusBranch: Combinatorial Branch-and-Bound for $\ell_0$ Neural Network Robustness Verification**, accepted at NeurIPS 2026. [[OpenReview](https://openreview.net/forum?id=VoRt2Q8ngL)] [[arXiv](#) (coming soon)]

<img src="fig_node_branching.svg#gh-light-mode-only" alt="Node branching" width="900">
<img src="fig_node_branching_dark.svg#gh-dark-mode-only" alt="Node branching" width="900">

<img src="fig_node_bound.svg#gh-light-mode-only" alt="Node bound" width="900">
<img src="fig_node_bound_dark.svg#gh-dark-mode-only" alt="Node bound" width="900">

## Prerequisites

First, clone the repo and `cd` into it
```bash
git clone --depth 1 https://github.com/simpl-group/FocusBranch.git
cd FocusBranch
```

### Download the Covering Design Database
Make sure you create the following folders
```bash
mkdir ../coverings
mkdir data/refinement_coverings/3
mkdir data/refinement_coverings/4
mkdir data/refinement_coverings/5
mkdir data/refinement_coverings/6
```

Download `refinement_coverings3.zip` ~ `refinement_coverings6.zip` into `../coverings`. The zip files can be obtained following the instructions from [`CoVerD`'s (one of our baseline) github repository](https://github.com/YuvShap/CoVerD).

```bash
unzip -q ../coverings/refinement_coverings3.zip -d data/refinement_coverings/3
unzip -q ../coverings/refinement_coverings4.zip -d data/refinement_coverings/4
unzip -q ../coverings/refinement_coverings5.zip -d data/refinement_coverings/5
unzip -q ../coverings/refinement_coverings6.zip -d data/refinement_coverings/6
```
### Other tools
Make sure CUDA is available and its version `>=12.2`. You can check your CUDA version via `nvcc --version`

Also make sure that your machine installed `gurobi>=12.0`  and have a valid `gurobi` license. You can check by running `grbprobe`.

If you don't have `uv`, download it
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Initialize the environment
```bash
uv venv --python 3.12.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python setup.py build_ext --inplace
python -m models.onnx2pt -anl --save
python -m data.db --init
sqlite3 data/data.db < data/sigma0.sql
```

## Usage
Then you can run our tool, for example, you can verify the $\ell_0$ robustness of afc1 on image #2 at $\epsilon=4$

```bash
python . --net afc1 -e 4 -s 2 -n 1 -to 1800 -g 4 -m 5 -rk 200 -rl 5000 -cb 4000 -sb 800 -pf 2 --demo -ut
```
if you want to verify afc1 for all 100 images with $\epsilon$ set to 5, and all the rest arguments stay default (e.g., number of GPU worker set to 4 and number of MILP worker set to 5), with results saved to local database
```bash
python . --net afc1 -e 5 -s 0 -tb 100 -to 1800 -g 4 -m 5 -rk 200 -rl 5000 -cb 4000 -sb 800 -pf 2
```

some command line arguments:
- `--net` choose a short nickname from afc1,afc2,afc3,fc4,bafc1,bfc1,bcv1, bcv2,...,bcv7. Note that afc4 in the paper refers to fc4 and afc5 refers to bafc1
- `-e` or `--eps`: perturbation budgets. choose from 1 to 6
- `-s` or `--start-idx`: the image index to start verificaiton, default to 0
- `-n` or `--num-images`: the total number of images to verify starting from the starting image index, default to 1
- `-to` or `--timeout`: the time limit in seconds for each image verification
- `-g` or `--num-gpus`: the number of GPU workers. Normally, it should not exceed the number of GPUs on your machine
- `-m` or `--num-milp`: the number of milp workers, default to 5
- `-rk`: the sample size of super nodes of each focus set size, **per GPU worker**
- `-rl`: the sample size of each level's concrete nodes, **per GPU worker**
- `-cb`: the batch size to bound concrete nodes
- `-sb`: the batch size to bound super nodes
- `-pf`: the pool factor: the super nodes of each focus set size are bounded once `-pf` x `-sb` of them are cached

some flags:
- `--demo`: If set, it bypass the local database and force a demo verification on a single image
- `-ut`: If set, we don't skip the verification until timedout, even if the estimated verification time is larger than timedout.
- `--concrete-only`: If set, run the ablation "w/o policy 1" (Section 4.4): super-node branching is disabled and verification always starts from the initial node $\Theta_0$, so only concrete nodes are branched and bounded, and no super node is sampled or bounded
- `--level-eps-only`: If set, run the ablation "w/o policy 2" (Section 4.4), which recovers the CoVerD+top-t-GP decomposition: concrete-node branching is disabled and verification always starts from level $\epsilon$ (requires $3 \le \epsilon \le 6$), without probing or concrete sampling. The two ablation flags are mutually exclusive; by default both are off and the starting level is chosen by the verification planner

For detailed command line args usage, please refer to `python . -h`.

#### If you want to use ABCrown as MILP solver, instead of Gurobi
1. build another venv with python 3.11, with all `requirements.txt` installed, activate it, run `python setup.py build_ext --inplace` again
2. clone [ABCrown repo](https://github.com/Verified-Intelligence/alpha-beta-CROWN) into another folder and `cd` into it, then run `uv pip install .` to install it locally
3. `cd` back to FocusBranch, rename the folder `util/` to eliminate conflict (and update all the references), for example rename to `utils_`
4. add command line argument `--solver abcrown` and run FocusBranch.

Note that abcrown can only be used for *fully connected networks*, and is roughly 10x slower than gurobi.


## Citation
If you use this code, please cite:

```
@inproceedings{
deng2026focusbranch,
title={FocusBranch: Combinatorial Branch-and-Bound for {$\ell_0$} Neural Network Robustness Verification},
author={Deng, Renwei and He, Yang and Li, Linyi and Wang, Yuepeng},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=VoRt2Q8ngL}
}
```