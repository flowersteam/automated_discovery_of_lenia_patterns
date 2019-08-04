# use conda
FROM continuumio/miniconda3

# set install directory
WORKDIR /install

# Copy the current directory contents into the destination directory
ADD ./autodisc/ /install/autodisc/

# create the autodisc environment
RUN conda env create -f /install/autodisc/conda_environment.yaml
RUN echo "source activate autodisc" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# install autodisc packages
RUN 	. /opt/conda/etc/profile.d/conda.sh && \
	conda activate autodisc && \
 	pip install /install/autodisc/

# set working directory
WORKDIR /lenia_experiments

# create script to start a jupyter notebook
COPY ./dockerfile/start-notebook.sh /usr/local/bin/start-notebook.sh  
RUN chmod +x /usr/local/bin/start-notebook.sh

# remove installation files
RUN rm -r /install

# expose ports for connection
# 8888 for jupyter notebook
EXPOSE 8888

ENTRYPOINT ["/bin/bash"]

# start the jupyter notebook server in the background
CMD ["-c", ". /opt/conda/etc/profile.d/conda.sh; conda activate autodisc; start-notebook.sh; bash"]

