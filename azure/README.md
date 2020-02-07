# DeepSpeed with Azure

This tutorial will help you get started running DeepSpeed on [Azure VMs](https://azure.microsoft.com/en-us/services/virtual-machines/), support for [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/) will be coming soon!

To help with launching Azure instances we suggest using the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/?view=azure-cli-latest). We have created several helper scripts to get you quickly started using DeepSpeed with Azure.
 * Install Azure CLI on your local box: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
 * Alternatively you can use the Azure in-browser shell: https://shell.azure.com/
 
 ## Create an SSH key
 Generate a SSH key that will be used across this tutorial to SSH into your VMs and between Docker containers. `ssh-keygen` is the recommended way of doing this. Our scripts assume your key is located inside the same directory as the Azure scripts.
 
 ## Azure Config JSON
 Our helper scripts depend on the following a configuration JSON for deployment and setup. We have provided a simple example JSON (see: [azure_config.json](azure_config.json)) that sets up a basic environment with two VMs, see the example below.
 ```json
 {
  "num_vms": 2,
  "location": "soutcentralus",
  "azure_sku": "Standard_NV6_Promo",
  "ssh_private_key": "id_rsa"
}
 ```
 
 ## Create Azure VMs
 [./create_vms.sh](create_vms.sh) will create VMs with the Azure SKU you choose and in the region you specify in your config JSON. Feel free to customize your JSON to your desired region/SKU. This step will take a few minutes to complete while it sets up all of your VMs on Azure.
  
 ## Setup VM environment to use DeepSpeed
 [./setup_vms.sh](setup_vms.sh) will generate the MPI-style hostfile and SSH config that all of your VMs will use so that your Docker containers can talk to one another after they are setup.
 
 ## Start the DeepSpeed docker container
 [./setup_docker.sh](setup_docker.sh) will pull the DeepSpeed docker image on all VMs and start a container instance in the background.
