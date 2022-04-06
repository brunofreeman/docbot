#!/bin/sh

if [ "${PWD##*/}" != "docbot" ]; then
  echo "usage: $0 must be run in docbot/, not ${PWD##*/}/"
  exit 1
fi

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <username>" >&2
  exit 1
fi

read -r -p "You are about to OVERWRITE your HPC docbot with your local version. Continue? (y/n) " yn

case $yn in
	[yY] )
	  echo "rsync -r --exclude=.git . ""$1""@login.hpc.caltech.edu:/home/""$1""/docbot"
	  rsync -r --exclude=.git . "$1"@login.hpc.caltech.edu:/home/"$1"/docbot;;
	[nN] ) echo "Overwrite aborted.";;
	* ) echo "Invalid response. Overwrite aborted.";;
esac
