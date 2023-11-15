#!/bin/bash

# SPDX-License-Identifier: Apache-2.0

DOCKER_IMAGE=nvcr.io/nvidia/pytorch:23.06-py3
DOCKER_Run_Name=pt_py3_torch

GPU_IDs=$1
#################################### check if parameters are empty
if [[ -z  $GPU_IDs ]]; then  #if no gpu is passed
    # for all gpus use line below
    GPU_IDs=all
    # for 2 gpus use line below
    # GPU_IDs=2
    # for specific gpus as gpu#0 and gpu#2 use line below
    # GPU_IDs='"device=1,2,3"'
fi

#################################### check if name is used then exit
docker ps -a|grep ${DOCKER_Run_Name}
dockerNameExist=$?
if ((${dockerNameExist}==0)) ;then
  echo --- dockerName ${DOCKER_Run_Name} already exist
  echo ----------- attaching into the docker
  docker exec -it ${DOCKER_Run_Name} /bin/bash
  exit
fi

echo -----------------------------------
echo starting docker for ${DOCKER_IMAGE} using GPUS ${GPU_IDs}
echo -----------------------------------

extraFlag="-it "
cmd2run="/bin/bash"

docker run --rm ${extraFlag} \
  --name=${DOCKER_Run_Name} \
  --net=host \
  --gpus ${GPU_IDs} \
  -e HOME=${HOME} \
  -v ${HOME}:${HOME} \
  -w ${HOME} \
  --shm-size=128g --ulimit memlock=-1 --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  ${cmd2run}
# --privileged -v /dev:/dev \
echo -- exited from docker image
