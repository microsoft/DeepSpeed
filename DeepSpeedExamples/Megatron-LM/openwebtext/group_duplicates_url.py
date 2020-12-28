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


def is_similar(jaccard_similarity):
    return (js >= 0.9)


if __name__ == '__main__':


    print('grouping duplicate urls ...')

    input = sys.argv[1]
    output = sys.argv[2]

    url_to_index = {}
    index_to_urls = []
    counter = 0
    start_time = time.time()
    with open(input, 'r') as f:
        for line in f:
            counter += 1
            myjson = json.loads(line)
            urls = []
            for main_url in myjson.keys():
                urls.append(main_url)
                for value in myjson[main_url]:
                    for other_url, js in value.items():
                        if is_similar(js):
                            urls.append(other_url)
            current_index = -1
            other_indices = set()
            for url in urls:
                if url in url_to_index:
                    if current_index == -1:
                        current_index = url_to_index[url]
                    elif current_index != url_to_index[url]:
                        other_indices.add(url_to_index[url])
            if current_index == -1:
                current_index = len(index_to_urls)
                index_to_urls.append(set())
            for url in urls:
                url_to_index[url] = current_index
                index_to_urls[current_index].add(url)
            for index in other_indices:
                for url in index_to_urls[index]:
                    index_to_urls[current_index].add(url)
                    url_to_index[url] = current_index
                index_to_urls[index] = None

            if counter % 100000 == 0:
                print(' > processed {} lines in {} seconds ...'.format(
                    counter, time.time() - start_time))


    total_remove = 0
    total_remain = 0
    for urls in index_to_urls:
        if urls is not None:
            if len(urls) > 1:
                total_remove += (len(urls) - 1)
                total_remain += 1
    print('out of {} urls, only {} are unique and {} should be removed'.format(
        total_remove+total_remain, total_remain, total_remove))

    with open(output, 'wb') as f:
        for i, urls in enumerate(index_to_urls):
            if urls is not None:
                if len(urls) > 1:
                    myjson = json.dumps({str(i): list(urls)},
                                        ensure_ascii=False)
                    f.write(myjson.encode('utf-8'))
                    f.write('\n'.encode('utf-8'))
