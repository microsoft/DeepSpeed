#!/bin/bash

num_vms=`az vm list | jq '. | length'`
username=deepspeed

for node_id in `seq 0 $((num_vms - 1))`; do
    ip_addr=`az vm list-ip-addresses | jq .[${node_id}].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
    addr=${username}@${ip_addr}
    ssh $addr "docker pull deepspeed/deepspeed:latest"
    ssh $addr "cd workdir/DeepSpeed; git pull; bash azure/start_container.sh"
done
