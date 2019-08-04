#!/bin/bash

# if the image does not exist, then create it
if [[ "$(docker images -q autodisc_image 2> /dev/null)" == "" ]]; then
	docker build -t autodisc_image -f dockerfile/Dockerfile .
fi

# start the container with a link to the source code
docker run -it --mount src="$(pwd)/experiments",target=/lenia_experiments,type=bind autodisc_image


