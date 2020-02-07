#!/bin/bash

docker_ssh_port=2222

num_vms=`az vm list | jq '. | length'`
username=azureuser
ssh_key=~/.ssh/id_rsa

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
    scp -o "StrictHostKeyChecking=no" ${ssh_key}* ${addr}:.ssh/
    scp -o "StrictHostKeyChecking=no" ${ssh_config} ${addr}:.ssh/
    ssh -o "StrictHostKeyChecking=no" ${addr} "sudo mkdir -p /job/; sudo chmod -R 777 /job; mkdir -p workdir"
    scp -o "StrictHostKeyChecking=no" ${hostfile} ${addr}:/job/
    ssh -o "StrictHostKeyChecking=no" ${addr} 'GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:microsoft/DeepSpeed.git workdir/DeepSpeed'
done
rm $hostfile $ssh_config
