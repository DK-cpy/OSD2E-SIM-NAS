import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test',type=str,default='test')
args = parser.parse_args()
c=args.test

print(c)