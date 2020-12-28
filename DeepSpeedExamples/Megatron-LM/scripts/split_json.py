"""
Takes a corpora of files (specified by `--input_files`) with json data separated
by newlines (loose json). Splits data into train.json, val.json, test.json files
under `output_dir`.

Note: This code has the potential to override files with the names 
train.json, val.json, test.json in `--output_dir`.
"""
import os
import argparse
import math
import random

parser = argparse.ArgumentParser('resplit loose json data into train/val/test')
parser.add_argument('--input_files', nargs='+', required=True,
                    help='whitespace separated list of input data files')
parser.add_argument('--output_dir', required=True,
                    help='output directory where to put files')
parser.add_argument('--test_percent', type=float, nargs='+', default=[0.05, 0],
                    help='percentage of available data to use for val/test dataset')
args = parser.parse_args()

def get_lines(filepath):
    lines = []
    with open(filepath, 'r') as f:
        for i, l in enumerate(f.readlines()):
            l = l.strip()
            lines.append(l)
    return lines

def get_splits(lines, line_counts):
    all_lines = []
    line_idx = []
    file_mappings = []
    for i, l in enumerate(lines):
        all_lines.extend(l)
        line_idx.extend(list(range(len(l))))
        file_mappings.extend([i]*len(l))

    indices = list(range(len(all_lines)))
    random.shuffle(indices)
    all_lines = [all_lines[idx] for idx in indices]
    line_idx = [line_idx[idx] for idx in indices]
    file_mappings = [file_mappings[idx] for idx in indices]
    
    splits = []
    mappings = []
    start = 0
    for end in line_counts:
        end += start
        splits.append(all_lines[start:end])
        mappings.append(format_mappings(line_idx[start:end], file_mappings[start:end]))
        start = end
    return splits, mappings

def format_mappings(line_idx, file_mappings):
    lines = []
    for m, l in zip(file_mappings, line_idx):
        lines.append(str(m).strip()+'\t'+str(l).strip())
    return lines


def get_filepaths(filepaths, output_dir):
    paths = []
    train_path = 'train.json'
    dev_path = 'dev.json'
    test_path = 'test.json'
    paths.append(os.path.join(output_dir, train_path))
    paths.append(os.path.join(output_dir, dev_path))
    paths.append(os.path.join(output_dir, test_path))
    return paths

def write_files(lines, mappings, filepaths):
    for l, m, path in zip(lines, mappings, filepaths):
        write_file(l, path)
        write_mapping_file(m, path)

def write_file(lines, path):
    print('Writing:', path)
    with open(path, 'w') as f:
        for l in lines:
            f.write(l+'\n')

def write_mapping_file(m, path):
    path = path+'.map'
    m = [get_mapping_header()]+m
    write_file(m, path)

def get_mapping_header():
    return 'file\tline #'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

lines = []

for filepath in args.input_files:
    _lines = get_lines(filepath)
    lines.append(_lines)

#calculate number of lines to use for each
line_counts = [len(l) for l in lines]
total_lines = sum(line_counts)
dev_percent = args.test_percent[0]
dev_lines = math.ceil(dev_percent*total_lines)
test_percent = 0
if len(args.test_percent)==2:
    test_percent=args.test_percent[1]
test_lines = math.ceil(test_percent*total_lines)
train_lines = total_lines-(test_lines+dev_lines)
normed_lines = [train_lines, dev_lines, test_lines]
normed_lines = [int(l) for l in normed_lines]


splits, mappings = get_splits(lines, normed_lines)
filepaths = get_filepaths(args.input_files, args.output_dir)
print('Writing output to:', filepaths)
write_files(splits, mappings, filepaths)

