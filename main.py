import json

from argparse import ArgumentParser

from Nram import NRam
from NRamContext import NRamContext
from GateFactory import GateFactory

if __name__ == "__main__":
    ap = ArgumentParser()

    ap.add_argument("--batch_size", "-bs",
                    dest="batch_size", type=int,
                    required=True, help="The examples to analyze", )
    ap.add_argument("--timesteps", "-t",
                    dest="timesteps", type=int,
                    required=True, help="The timesteps that the NRAM must execute", )
    ap.add_argument("--max_int", "-mi",
                    dest="max_int", type=int,
                    required=True, help="The dimension of the number set", )
    ap.add_argument("--task_type", "-tt",
                    dest="task_type", type=str,
                    required=True, help="The task to execute",
                    choices=["task_copy", "task_access"], )
    ap.add_argument("--network", "-n",
                    dest="network", type=str,
                    required=True, help="The file where resides the network", )
    ap.add_argument("--gates", "-g",
                    dest="gates", type=str, nargs="*",
                    required=True, help="The list of gates to use",
                    choices=["read", "zero", "one", "two", "inc", "add", "sub",
                             "dec", "lt", "let", "eq", "min", "max", "write"])

    args = ap.parse_args()

    with open(args.network) as f:
        test = json.load(f)

    network = test["network"]


    context = NRamContext(
        batch_size=args.batch_size,
        max_int=args.max_int,
        timesteps=args.timesteps,
        task_type=args.task_type,
        network=network,
        gates=[ GateFactory.create(g)for g in args.gates ]
    )

    nram = NRam(context)

    nram.execute()
