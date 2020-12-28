"""
Usage:
python scripts/presplit_sentences_json.py <original loose json file> <output loose json file>
"""

import sys
import json

import nltk

nltk.download('punkt')

input_file = sys.argv[1]
output_file = sys.argv[2]

line_seperator = "\n"

with open(input_file, 'r') as ifile:
  with open(output_file, "w") as ofile:
    for doc in ifile.readlines():
      parsed = json.loads(doc)
      sent_list = []
      for line in parsed['text'].split('\n'):
          if line != '\n':
              sent_list.extend(nltk.tokenize.sent_tokenize(line))
      parsed['text'] = line_seperator.join(sent_list)
      ofile.write(json.dumps(parsed)+'\n')
