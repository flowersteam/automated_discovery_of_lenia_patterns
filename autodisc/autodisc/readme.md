
# Introduction

Autodisc is a framework for automated discovery in simulated and physical systems.
The framework provides algorithms to explore such systems.



# Table of Contents

* [Team Members](#team-members)

* [Requirements](#requirements)

* [Documentation](#documentation)

* [Development Notes](#dev_notes)



# <a name="team-members"></a>Team Members

* [Chris Reinke](http:www.scirei.net) <chris.reinke@inria.fr>
* Mayalen Etcheverry <mayalen.etcheverry@inria.fr>
* [Pierre-Yves Oudeyer](http://www.pyoudeyer.com/) <pierre-yves.oudeyer@inria.fr>



# <a name="requirements"></a>Requirements

An anaconda environment is provided with the file **autodisc_conda_env.yaml** that has all required packages.
To install the environment **conda** is required.
Then use the following command to create the environment:

`conda env create -n autodisc -f autodisc_conda_env.yaml`

Additionally the following python packages are required:
* [neat-python](https://github.com/CodeReclaimers/neat-python) 


# <a name="documentation"></a>Documentation

## <a name="dev_notes"></a>Results

The following data is usually saved as results from an exploration:

* **Exploration**:
    * id - optional integer identifier of the exploration
    * name - optional name string
    * descr - optional description string for the exploration
    * seed - random seed at the beginning of the exploration
conda env create -name test --file autodisc_conda_env2.yaml 

    * system_name - name of the system that was explored
    * system_parameters - parameters of the system that are the same for all exploration runs 
    * statistics - AttrDict with the statistics over the exploration (keys: names, values: data)
    * **runs**: - dictionary with single exploration runs (keys: ids, values: data)  
        * id - integer identifer of the run
        * name - optional name string
        * seed - start seed of the run
        * run_parameters - parameters for the system 
        * **observations**: - AttrDict with:
            * states - observed states
            * timepoints - timepoints of the observed states
        * statistics - AttrDict with the statistics over the run (keys: names, values: data)


# <a name="dev_notes"></a>Development Notes

General:
* To add the package for development purposes to the python path see the following discussion.
The solution with the .pth files works well:
    * https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath

PyCharm Notes:
* To show the progress bars in the console output activate the "Emulate terminal in output console" option in Run/Debug Configuration  