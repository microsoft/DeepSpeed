#!/bin/bash

location=southcentralus
azure_sku=Standard_NV6_Promo
resource_group=DeepSpeedRG_$location

az group create --name ${resource_group} --location $location

ssh_key=~/.ssh/id_rsa.pub

base_vm_name=deepspeed
vm_image="nvidia:ngc_azure_17_11:ngc_gpu_cloud_19_11_3:19.11.3"

num_vms=2

#az vm image terms accept --urn ${vm_image}
for i in `seq 0 $(( num_vms - 1))`; do
    vm_name=${base_vm_name}_$i
    echo "creating $vm_name"
    az vm create \
      --resource-group ${resource_group} \
      --name ${vm_name} \
      --image ${vm_image} \
      --admin-username deepspeed \
      --size ${azure_sku} \
      --ssh-key-values ${ssh_key}
done
