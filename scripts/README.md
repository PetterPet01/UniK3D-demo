## Training

We provide the `train.py` script that allows to load the dataset, initialize and start the training. From the root of the repo:

```bash
export REPO=`pwd`
export PYTHONPATH=${REPO}:${PYTHONPATH}

# Adapt all this to your setup
export TMPDIR="/tmp"
export TORCH_HOME=${TMPDIR}
export HUGGINGFACE_HUB_CACHE=${TMPDIR}
export WANDB_HOME=${TMPDIR}
export DATAROOT=<where-you-stored-the-hdf5>


export MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))
if [ $NNODES -gt 1 ]; then
    export MASTER_PORT=29400
fi

# this is the config will be used
export CFG="config_vitl.json"
```

If you are on a machine without SLURM you can run the following:
```bash
# make the following input-dependent for multi-node
export NNODES=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES="0" # set yours

export GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
echo "Start script with python from: `which python`"
torchrun --rdzv-backend=c10d --nnodes=${NNODES} --nproc_per_node=${GPUS} --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} ${REPO}/scripts/train.py --config-file ${REPO}/configs/${CFG} --distributed
```

If you system has SLURM, all the information will be set by the scheduler and you have to run just:
```bash
srun -c ${SLURM_CPUS_PER_TASK} --kill-on-bad-exit=1 python -u ${REPO}/scripts/train.py --config-file ${REPO}/configs/${CFG} --master-port ${MASTER_PORT} --distributed
```


### Datasets

We used both image-based and sequence-based dataset. The `ImageDataset` class is actually for legacy only as we moved image-based dataset to be "dummy" single-frame sequences.<br>
We [provide two example dataset to get familiar to the pipeline and structure, namely iBims-1 and Sintel](https://drive.google.com/drive/folders/1FKsa5-b3EX0ukZq7bxord5fC5OfUiy16?usp=sharing), image- and sequence-based, respectively.<br>
You can adapt the data loading and processing to your example; however, you will need to keep the same interface for the model to be consisten and train "out-of-the-box" the model.<br>


### Additional dependencies

We require chamfer distance for the evaluation, you can compile the knn operation under `ops/knn`: `bash compile.sh` from the directory `$REPO/unik3d/ops/knn`. Set the correct `export TORCH_CUDA_ARCH_LIST`, according to the hardware you are working on.
For training and to perform augmentation, you can use `camera_augmenter.py`; however the splatting requires you to install operations by cloning and installing from `github.com/hperrot/splatting`.