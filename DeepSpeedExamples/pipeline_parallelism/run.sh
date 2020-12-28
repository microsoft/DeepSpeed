#!/bin/bash

deepspeed train.py --deepspeed_config=ds_config.json -p 2 --steps=200
