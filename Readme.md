# Automated Discovery of Patterns in Lenia

04.08.2019, Version 1

Python source code of experiments and data analysis for the paper: 

**Intrinsically Motivated Exploration for Automated Discovery of Patterns in Morphogenetic Systems**  
[Chris Reinke](http://www.scirei.net), [Mayalen Echeverry](https://mayalenetcheverry.com), [Pierre-Yves Oudeyer](http://www.pyoudeyer.com)

More information: https://automated-discovery.github.io

Contact us for questions or comments: chris.reinke@inria.fr


## Installation

First, clone this repository with
```
git clone https://github.com/flowersteam/automated_discovery_of_lenia_patterns.git
```
Note that you need the Git Large File Storage (LFS) extension (https://git-lfs.github.com/) for downloading the trained VAE models (optional).

We provide 2 options to install and run the experiments:
1. Docker environment [Recommended]
2. Installation into a conda environment


### Docker environment

We provide a docker environment that has everything installed to run the experiments. Docker can be understood as a virtual machine.

1. Install [Docker](https://docs.docker.com/install/overview/) (e.g. on [Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) )

2. Create the docker image and start the environment either by using the provided script: `./start_docker_environment.sh`

   Alternatively, you can do the steps yourself by:
   * Create the image:  
      `docker build -t autodisc_image -f dockerfile/Dockerfile .`
   * Start the environment:  
      `docker run -it --mount src="$(pwd)/experiments",target=/lenia_experiments,type=bind autodisc_image`

### Installation into a conda environment

Conda is a package and virtual environment managing tool. We will use it to create an virtual environment in which the experiments are running and to hold all required packages. We recommend Ubuntu 16.04 LTS as OS, because we used it as our development platform. 

To install environment and all required packages:

1. Install [conda](https://www.anaconda.com/) (e.g. on [Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart))  

2. Creation of the *autodisc* virtual environment:  
   `conda env create -n autodisc -f ./autodisc/conda_environment.yaml`

3. Installation of the *autodisc* package which has all required algorithms, systems and tools:  
   1. Start *autodisc* environment: `conda activate autodisc`  
   2. Install packages: `pip install ./autodisc/`



## Description & Usage

The experiments are conducted in 4 phases. For each phase exists a subdirectory in the *experiments* directory:
1. *pre_train_imgep_pgl_goalspace*: Optional training of the VAEs that are used as encoders for the goal space of the IMGEP-PGL. 
2. *explorations*: All exploration experiments and their data analysis.
3. *post_train_analytic_behavior_space*: Optional training of the VAE that learns features for the analytic behavior space from patterns identified during the experiments. 
4. *post_train_analytic_parameter_space*: Optional training of the VAE that learns features for the analytic parameter space from start patterns used during the experiments.

**Analysis and plotting of results**

After the experiments are performed the results can be plotted using [Jupyter](https://jupyter.org/) notebooks. For this purpose each phase subdirectory has an *analyze* directory in which the notebooks are located. 

If the Docker environment is used, a Jupyter notebook server is running in the background. It can be accessed via the link given on the command shell after the docker environment is started.

If a conda environment is used, start the Jupyter notebook while being in the *autodisc* environment:   
	`conda activate autodisc`  
	`jupyter notebook`


### 1. pre_train_imgep_pgl_goalspace (optional) 

Optional training of the VAE encoders for the goal spaces of the IMGEP-PGL approaches. See Section 5 of the paper's Supplementary Material for more information. The step is optional because we included the trained VAEs in the source code. Nonetheless, the training can be redone if wanted.

**Content**:

 * *analyze*: Jupyter notebooks to plot training results.
 * *data*: Dataset for the training of the VAEs. 
 * *code*: Template code from which the training code is generated.
 * *experiments*: Code for the training.
 * *experiment_configurations.ods*: Configuration of the experiments.
 * *generate_experiments.sh*: Generates the code for the training procedures. 

**Usage**:

1. (Optional) If changes were made to the *experiment_configurations.ods* file then generate the code for the training using the script:  
   `./generate_experiments.sh`
1. Start the training for each VAE by running the scripts *run_training.py* in the subfolders of the *experiments* folder, for example:  
`cd experiments/000017/`  
`python run_training.py`


### 2. explorations

Code to run the exploration experiments and to analyze the results. 

We are using the *exputils* package in the *autodisc* framework to generate the code for the experiments from template files. The Libreoffice spreadsheet *experiment_configurations.ods* holds the configuration for all experiments. For the source code the experiment files have been already generated. However, it is possible to change configurations in the *experiment_configurations.ods* and use the *generate_experiments.sh* to change the existing configuration or to generate new experiments. 
Note: In case the number of repetitions is reduced then the existing repetition folders will not be deleted. In this case all experiment directories under *experiments* should be removed and they are newly generated.

For each experiment, which represents the exploration with a specific algorithm/configuration, several repetitions are performed with different random seeds. For each repetition individual code files are generated. This allows to run repetitions and experiments in parallel for example on a cluster. The *exputils* package provides tools to start experiments and repetitions in parallel. We added scripts for this purpose for clusters using the [SLURM](https://slurm.schedmd.com/overview.html) job manager.

**Content**:

 * *analyze*: Jupyter notebooks to plot experiment results.
 * *code*: Template code from which the experiments are generated.
 * *experiments*: Code of experiments.
 * *experiment_configurations.ods*: Configuration of the experiments.
 * *generate_experiments.sh*: Generates the experiment code. 
 * *run_local_\*.sh*: Scripts to run explorations and statistic calculations on a PC.
* *run_slurm_\*.sh*: Scripts to run explorations and statistic calculations on a cluster with SLURM.

**Usage**:
Experiments run in 2 phases. First, by running the explorations with several repetitions per algorithm/configuration. Second, by computing the statistics over the repetitions for each algorithm/configuration. Several  scripts to calculate statistics exist.

1. (Optional) If the configuration of experiments is changed in the *experiment_configurations.ods*, then the experiment files need to be regenerated by:  
   `./generate_experiments.sh`
   
1. Starting the experiments. Experiments can be started individually by executing the *run_experiment.py* in each repetition directory. 
   
   Alternatively, all experiments can be started by using:  
      `./run_local_experiments.sh`  (or if on a SLURM cluster: `./run_slurm_experiments.sh`)  
   
   **Notice**: The experiments usually need a very long time (8h per repetition). We recommend therefore to run them on a cluster. If this is not possible then the number of repetitions in the *experiment_configurations.ods* should be reduced and only individual experiments should be run.

3. Calculating the general statistics. Calculation of statistics can be started individually by executing the *calc_statistics_over_repetitions.py* in each experiment directory and the *calc_statistics_per_repetition.py* in each repetition directory. 
   Alternatively, all statistics can be calculated by:
   `./run_local_calc_statistics_over_repetitions.sh`  
         (or if on a SLURM cluster: `./run_slurm_calc_statistics_over_repetitions.sh`)  
   `./run_local_calc_statistics_per_repetition.sh`  
         (or if on a SLURM cluster: `./run_slurm_calc_statistics_per_repetition.sh`) 

4. Calculating the analytic behavior space and analytic parameter space. See Section 3 of the paper's Supplementary Material for more information. These calculations depend on the VAEs trained in Phase 3 (*post_train_analytic_behavior_space*) and 4 (*post_train_analytic_parameter_space*). These phases are optional because the VAEs from the original experiments are included in the source code. Thus, if these Phases are skipped, then this step can be immediately executed. Otherwise, it is executed after Phase 4 and 5.

   Calculation of the spaces can be started individually by executing the *calc_statistic_space_representation.py* in each experiment directory. 
   Alternatively, all statistics can be calculated by:  
   `./run_local_calc_statistic_space_representation.sh`  
         (or if on a SLURM cluster: `./run_slurm_calc_statistic_space_representation.sh` ) 

5. After the statistics are calculated, the Jupyter notebooks in the *analyze* directory can be used to view the results. 
The *make_pdf_figures_\*.ipynb* notebooks provide source code to generate all the figures shown in the paper.
The *interactive_visualisation_obtained_goalspaces.ipynb* notebook provides an interactive viewer tool to explore the found patterns during the different IMGEP experiments. A video with demonstration of the interface can be found on the website.

### 3. post_train_analytic_behavior_space (optional) 

Optional training of the VAE used to encode the final patterns for the analytic behavior space. See Section 3 of the paper's Supplementary Material for more information. The training is optional because the trained VAE from the original experiments is included in the source code. 

**Content**:

* *analyze*: Jupyter notebooks to plot training results.
* *data*: Scripts to collect the dataset of identified patterns during all explorations.
* *training*: Scripts to train the VAE. Contains already the VAE trained for the results for the paper.

**Usage**:

1. Before the VAE is trained, a dataset needs to be collected over all patterns identified in the experiments.  
   `cd data`  
   `python collect_data`
2. Train the VAE on the collected data:  
   `cd ../training`  
   `python run_training.py`
3. After the VAE is trained, it can be used to calculate the analytic behavior space. See step 4 of the usage for the *explorations*.

### 4. post_train_analytic_parameter_space (optional) 

Optional training of the VAE used to encode the initial states of Lenia for the analytic parameter space. See Section 3 of the paper's Supplementary Material for more information. The training is optional because the trained VAE from the original experiments is included in the source code. 

**Content**:

- *analyze*: Jupyter notebooks to plot training results.
- *data*: Scripts to collect the dataset of initial patterns used during all explorations.
- *training*: Scripts to train the VAE. Contains already the VAE trained for the results for the paper.

**Usage**:

1. Before the VAE is trained, a dataset needs to be collected over all patterns identified in the experiments.  
   `cd data`  
   `python collect_data`

2. Train the VAE on the collected data:  
   `cd ../training`  
   `python run_training.py`

3. After the VAE is trained, it can be used to calculate the analytic parameter space. See step 4 of the usage for the *explorations*.

   


## Notes

* Runtime warnings during the experiments can be ignored.
* To run experiments on a cluster it might be necessary to include in most script files, e.g. *run_experiment.slurm* of the exploration experiments an activation of the *autodisc* conda environment. See the commented lines in the scripts.
* Some names for data files and variables in the code differ from the names used in the paper:
   * The analytic behavior space is often just termed *statistic space*.
   * The measurement of centeredness of a pattern is called *mass_distribution* in the source code.
