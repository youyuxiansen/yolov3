import torch
import struct
import sys
import argparse
import os
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output', help='Output (.wts) file path (optional)')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args.weights, args.output


pt_file, wts_file = parse_args()

# Initialize
device = select_device('cpu')
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        print(k)
        # if 'anchor_grid' not in k:
        #     continue
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
