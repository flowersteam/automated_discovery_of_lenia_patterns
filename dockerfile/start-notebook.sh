#!/bin/bash

# start jupyter notebook server in background
# do not use a token or password
exec jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' /lenia_experiments &> /dev/null &

# print the ip address for the server
DOCKERIP=$(ip addr | grep global | grep -E -o "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)" | head -n1)

echo "Jupter Notebook can accessed in a webbrowser via: http://$DOCKERIP:8888" 

