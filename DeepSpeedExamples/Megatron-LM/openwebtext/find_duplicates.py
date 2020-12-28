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


import itertools
import json
from lsh import cache, minhash
import time
import sys


# This function is adapted from:
#   https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram]
               for head in range(0, len(text) - char_ngram))


# This function is adapted from:
#  https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


if __name__ == '__main__':

    print('finding possible duplicate content ...')

    input = sys.argv[1]
    output = sys.argv[2]

    hasher = minhash.MinHasher(seeds=100, char_ngram=5, hashbytes=4)
    lshcache = cache.Cache(bands=10, hasher=hasher)

    counter = 0
    url_doc = {}
    start_time = time.time()
    with open(input, 'r') as f:
        for line in f:
            try:
                myjson = json.loads(line)
                url = myjson['url']
                text = myjson['text']
                counter += 1
                url_doc[url] = text
                lshcache.add_fingerprint(hasher.fingerprint(text), url)
            except Exception as e:
                print('Error:', e)
            if counter % 10000 == 0:
                print(' [read]> processed {} documents in {:.2f} seconds ...'.
                      format(counter, time.time() - start_time), flush=True)

    counter = 0
    start_time = time.time()
    deduped = 0
    with open(output, 'wb') as f:
        for b in lshcache.bins:
            for bucket_id in b:
                if len(b[bucket_id]) > 1:
                    items = list(b[bucket_id])
                    main_url = items[0]
                    main_dhingles = shingles(url_doc[main_url])
                    remove_urls = []
                    for i in range(1, len(items)):
                        counter += 1
                        other_url= items[i]
                        other_shingles = shingles(url_doc[other_url])
                        try:
                            jaccard_sim = jaccard(main_dhingles, other_shingles)
                        except Exception as e:
                            print('Error:', e)
                        if jaccard_sim > 0.5:
                            remove_urls.append({other_url: jaccard_sim})
                            deduped += 1
                        if counter % 10000 == 0:
                            print(' [write]> processed {} documents in {:.2f} '
                                  'seoncds and deduped {} documents ...'.
                                  format(counter, time.time() - start_time,
                                         deduped), flush=True)
                    if len(remove_urls) > 0:
                        myjson = json.dumps({main_url: remove_urls},
                                            ensure_ascii=False)
                        f.write(myjson.encode('utf-8'))
                        f.write('\n'.encode('utf-8'))

    print('done :-)')
