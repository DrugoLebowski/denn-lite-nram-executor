# Standard
import json
from argparse import ArgumentParser

# Project
from Nram import NRam
from NRamContext import NRamContext
from GateFactory import GateFactory

if __name__ == "__main__":
    ap = ArgumentParser()


    ap.add_argument("file", type=str,
                    help="The file where resides the configuration of NRAM", )
    ap.add_argument("--batch_size", "-bs",
                    dest="batch_size", type=int,
                    help="The examples to analyze", )
    ap.add_argument("--timesteps", "-t",
                    dest="timesteps", type=int,
                    help="The timesteps that the NRAM must execute", )
    ap.add_argument("--max_int", "-mi",
                    dest="max_int", type=int,
                    help="The dimension of the number set", )
    ap.add_argument("--task_type", "-tt",
                    dest="task_type", type=str,
                    help="The task to execute",
                    choices=["task_copy", "task_access"], )

    args, leftovers = ap.parse_known_args()

    with open(args.file) as f:
        test = json.load(f)
        test_args = test["arguments"]

    NRam(NRamContext(
        batch_size=args.batch_size if args.batch_size is not None else 2,
        max_int=args.max_int if args.max_int is not None else test_args["max_int"],
        timesteps=args.timesteps if args.timesteps is not None else test_args["time_steps"],
        task_type=args.task_type if args.task_type is not None else "task_%s" % test_args["task"],
        network=test["network"],
        gates=[ GateFactory.create(g) for g in test_args["gates"] ]
    )).execute()
