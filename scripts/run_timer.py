import argparse
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--time', '-t', type=int, default=15)
parser.add_argument('command', type=str)

args = parser.parse_args()


process = subprocess.Popen(args.command.split(' '))
print('Running command: {}'.format(args.command))
print('Waiting for {} seconds'.format(args.time))
time.sleep(args.time)

print('Terminating process')
process.terminate()