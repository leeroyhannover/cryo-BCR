
# for python CLI entry point to be available from arbitrary location
export PATH=/path/to/cryo-BCR/bin:${PATH}
# for CUDA/GPU-based devices recongnition
export LD_LIBRARY_PATH=/path/to/micromamba/envs/cryobcr-env/lib/python3.9/site-packages/nvidia/cudnn/lib/:${LD_LIBRARY_PATH}
