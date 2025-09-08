assert __name__ == "__main__"
if __package__ is None or __package__ == '':
    from covering_generator import CoveringConstructor
else:
    from . import CoveringConstructor

import argparse
from multiprocessing.connection import Client

parser = argparse.ArgumentParser(description='Generate covering blocks for L0 robustness verification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--q', type=int, required=True, help='the order of the finite field')
parser.add_argument('--m', type=int, required=True, help='the dimension of the vector space')
parser.add_argument('--t', type=int, required=True, help='the size of subsets to be covered')
parser.add_argument('--chosen_indices', type=int, nargs='+', required=True, help='the indices of chosen points (0-indexed)')
parser.add_argument('--num-workers', type=int, default=1, help='Total number of workers')
parser.add_argument('--worker-id', type=int, default=0, help='The id of this worker')
parser.add_argument('--port', type=int, required=True, help='The port main verification process is listening on')
args = parser.parse_args()

covering_constructor = CoveringConstructor(q=args.q, m=args.m, t=args.t, chosen_indices=args.chosen_indices)
with Client(('localhost', args.port)) as conn:
    for block_index, block in enumerate(covering_constructor.construct_induced_covering()):
        if block_index % args.num_workers == args.worker_id:
            conn.send(block.tolist())

    conn.send('done')
    message = conn.recv()
    assert message == 'done'