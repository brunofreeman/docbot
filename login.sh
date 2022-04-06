#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <username>" >&2
  exit 1
fi

echo "ssh ""$1""@login.hpc.caltech.edu"
ssh "$1""@login.hpc.caltech.edu"
