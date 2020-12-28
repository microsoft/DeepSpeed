# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import sys
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=".",
        help="path where all the json files are located")

    parser.add_argument("--output_file", type=str, default="merged_output.json",
        help="filename where the merged json should go")

    args = parser.parse_args()

    json_path = args.json_path
    out_file = args.output_file

    json_files = glob.glob(json_path + '/*.json')

    counter = 0

    with open(out_file, 'w') as outfile:
        for fname in json_files:
            counter += 1

            if counter % 1024 == 0:
                print("Merging at ", counter, flush=True)

            with open(fname, 'r') as infile:
                for row in infile:
                    each_row = json.loads(row)
                    outfile.write(row)


    print("Merged file", out_file, flush=True)


