#!/bin/bash

SSH_LIST=(35.208.92.231 35.206.107.250 35.208.115.196)

for ip in "${SSH_LIST[@]}"; do
  rsync -avzh --progress /job/hostfile yunmokoo@"$ip":/home/yunmokoo
done

