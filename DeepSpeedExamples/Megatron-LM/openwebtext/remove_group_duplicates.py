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


import json
import time
import sys


if __name__ == '__main__':

    url_filename = sys.argv[1]
    data_filename = sys.argv[2]
    output_filename = sys.argv[3]

    urls = set()
    with open(url_filename, 'r') as f:
        for line in f:
            myjson = json.loads(line)
            for key in myjson:
                this_urls = myjson[key]
                for i in range(1, len(this_urls)):
                    urls.add(this_urls[i])
    print('will be removing {} urls'.format(len(urls)), flush=True)

    written_docs = 0
    removed_docs = 0
    removed_chars = 0
    start_time = time.time()
    with open(output_filename, 'wb') as fout:
        with open(data_filename, 'r') as fin:
            for line in fin:
                try:
                    myjson = json.loads(line)
                    url = myjson['url']
                    if url in urls:
                        print('removing', myjson)
                        removed_docs += 1
                        removed_chars += len(myjson['text'])
                        continue
                    myjson = json.dumps(myjson, ensure_ascii=False)
                    fout.write(myjson.encode('utf-8'))
                    fout.write('\n'.encode('utf-8'))
                    written_docs += 1
                    if written_docs % 10000 == 0:
                        print(' [PROCESSED] time (s): {:.2f} | written: {} '
                              '| removed: {} (char: {})'.format(
                                  time.time() - start_time,
                                  written_docs, removed_docs, removed_chars))
                except Exception as e:
                    print('[SKIPPING]', line, e)

    print(' [PROCESSED] time (s): {:.2f} | written: {} '
          '| removed: {} (char: {})'.format(
              time.time() - start_time,
              written_docs, removed_docs, removed_chars))
    print('done :-)')
