#!/bin/bash

SSH_LIST=(35.208.173.14 35.208.115.196 35.208.92.231)

for ip in "${SSH_LIST[@]}"; do
  rsync -avzh --progress /job/hostfile yunmokoo@"$ip":/job/hostfile
done

