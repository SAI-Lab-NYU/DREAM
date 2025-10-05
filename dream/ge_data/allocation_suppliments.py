import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='ge_data/llava_vicuna_7B')
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

s = 0
e = 1000 - 1

gpus = [[0]]

num_p = len(gpus)
outdir = '{}/llava_{}_{}_mubf16'.format(args.outdir, 0, e)

def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)

if not os.path.exists(outdir):
    os.makedirs(outdir)

data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    gpu_index = gpus[i]
    gpu_index_str = ' '.join(map(str, gpu_index))
    command = "python ge_data/ge_data_suppliments.py --start={} --end={} --index={} --gpu_index {} --outdir {}".format(start, end, index, gpu_index_str, outdir)
    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
