#!/bin/sh

if [ "${HOSTNAME}" = "login1.cm.cluster" ]; then
  echo "usage: $0 must be run locally, not on ${HOSTNAME}"
  exit 1
fi

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <username>" >&2
  exit 1
fi

echo "ssh ""$1""@login.hpc.caltech.edu"
ssh "$1""@login.hpc.caltech.edu"
