#!/bin/bash

num_vms=`az vm list | jq '. | length'`
username=deepspeed
ssh_key=id_rsa

for node_id in `seq 0 $((num_vms - 1))`; do
    ip_addr=`az vm list-ip-addresses | jq .[${node_id}].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
    addr=${username}@${ip_addr}
    ssh -i ${ssh_key} $addr "docker pull deepspeed/deepspeed:latest"
    ssh -i ${ssh_key} $addr "cd workdir/DeepSpeed; git pull; git checkout jeffra/azure; bash azure/start_container.sh"
done
