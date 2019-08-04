# Automated Discovery Tool Package by Flowers, INRIA

Version: **0.0.1** (alpha)

This Python package provides algorithms, environments and tools for the automated discovery of new behaviors, patterns and dynamics in simulated physical and chemical systems. It is developed by the [Flowers](https://flowers.inria.fr) research team of [Inria](https://www.inria.fr/en/) Bordeaux.

For more information see: https://automated-discovery.github.io




## <a name="team-members"></a>Team Members

* [Chris Reinke](http:www.scirei.net) (<chris.reinke@inria.fr>)
* Mayalen Etcheverry (<mayalen.etcheverry@inria.fr>)
* [Pierre-Yves Oudeyer](http://www.pyoudeyer.com/) (<pierre-yves.oudeyer@inria.fr>)




## <a name="installation"></a>Installation

### Requirements

The autodisc package has several dependencies on other packages which are all available in Anaconda. [Anaconda](https://www.anaconda.com/) is a platform that provides many ML libraries and builds upon [Conda](https://conda.io/en) which is a package management tool. We recommend the usage of a conda environment to use autodisc.  

The file *conda_environment.yaml* provides a list of all required packages and allows to set up a conda environment.

Instructions for the installation of (ana)conda: https://www.anaconda.com/distribution/

After conda is installed the following command installs the conda environment with all required dependencies::

`conda env create -n autodisc -f conda_environment.yaml`

Additionally the following python packages are required which are not managed by anaconda:
* [neat-python](https://github.com/CodeReclaimers/neat-python)

Please install them in the new conda environment following the installation steps on their websites.

### Installation of the autodisc packages

To install the packages in the new autodisc environment use:

`conda activate autodisc`

`pip install .`


## <a name="documentation"></a>Usage

The autodisc framework is currently in an alpha version resulting in a sparse documentation. We recommend reviewing the code of experiments that use autodisc provided by us, studying the added demonstrations and the test cases.

The framework consists of 2 packages:

* <u>autodisc</u>: Core files needed for experiments, algorithms, systems, data and visualization tools.
* <u>exputil</u>: Experiment utils that allow to generate experiments with different configurations and with several repetitions. Provides tools to start the experiments in parallel, for example on a cluster.

### autodisc

See the demonstrations for how to create and start an experiment:

* autodisc/demos/lenia:
  * *animal_exploration.py*: Explores manually identified animals for the Lenia system.
  * *random_exploration.py*: Performs a random exploration in the Lenia system.
  * *goalspace_exploration*: Performs IMGEP explorations for the Lenia system. Use the *generate_experiments.sh* script to generate the experiments and the *run_experiments.sh* script to run them. The jupyter notebooks in the *analyze* folder can be used to view the results.
* autodisc/demos/cppn_evolution:
  * *cppn_evolution.py*: Performs and visualizes a evolution of Lenia patterns via CPPNs.

### exputils

The exputils can be used to generate experiments with different parameters and several repetitions. Please see the demo *autodisc/demos/lenia/goalspace_exploration* for an example and its usage.
