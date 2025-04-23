Run run_path_extraction.sh on HiPerGator (HPG) through slurm, specifying the directory of images to split into 224x224 tissue-containing patches.

Then run image_reconstruction.sh on HPG through slurm, specifying the output folder from the previous script. This will rebuild the patches into 1792x1792 individual samples that are ready for training.

Both scripts make use of a conda python virtual environment, as the HPG module system does not allow for certain python packages to be loaded together. This environment can be found at /blue/vabfmc/data/working/tannergarcia/DermHisto/conda/envs/image_splitting/ or built from scratch by loading the conda module, creating a new environment, then interacting with the python executeable it creates.