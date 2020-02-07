#!/bin/bash

docker_ssh_port=2222
username=deepspeed
ssh_key=~/.ssh/id_rsa

num_vms=`az vm list | jq '. | length'`
first_ip_addr=`az vm list-ip-addresses | jq .[0].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
num_slots=`ssh ${username}@${first_ip_addr} 'nvidia-smi -L | wc -l'`

hostfile=hostfile
ssh_config=config
for node_id in `seq 0 $((num_vms - 1))`; do
    private_ip_addr=`az vm list-ip-addresses | jq .[${node_id}].virtualMachine.network.privateIpAddresses[0] | sed 's/"//g'`
    echo "worker-${node_id} slots=${num_slots}" >> hostfile
    echo "Host worker-${node_id}
    HostName ${private_ip_addr}
    Port ${docker_ssh_port}
    " >> ${ssh_config}
done

ssh_args='-o "StrictHostKeyChecking=no"'
for node_id in `seq 0 $((num_vms - 1))`; do
    ip_addr=`az vm list-ip-addresses | jq .[${node_id}].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
    addr=${username}@${ip_addr}
    echo "copying ssh keys, ssh config, hostfile to worker-${node_id}"
    scp ${ssh_key}* ${addr}:.ssh/
    scp ${ssh_config} ${addr}:.ssh/
    ssh ${addr} "sudo mkdir -p /job/; sudo chmod -R 777 /job; mkdir -p workdir"
    scp ${hostfile} ${addr}:/job/
    ssh ${addr} 'GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:microsoft/DeepSpeed.git workdir/DeepSpeed'
done
rm $hostfile $ssh_config
