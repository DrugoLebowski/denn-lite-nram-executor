# Standard
import os
import json
from argparse import ArgumentParser

# Project
from App import App
from NRam import NRam
from NRamContext import NRamContext
from factories.GateFactory import GateFactory
from util import exists_or_create


if __name__ == "__main__":
    exists_or_create(App.get("images_path"))

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
    ap.add_argument("--debug", "-d",
                    dest="debug", nargs="?", const=True, default=False,
                    help="Write out the debug of NRAM", )
    ap.add_argument("--print_circuits", "-pc",
                    dest="print_circuits", nargs='?', const=True, default=False,
                    help="Draw the circuit of each timestep to a file with a progressive numeration [e.g. <filename>.1.1.png, ..., <filename>.S.T.png, where S is the sample and T is the timestep]",)
    ap.add_argument("--print_memories", "-pm",
                    dest="print_memories", nargs='?', const=True, default=False,
                    help="Draw an image where the initial memories modified by the circuits are compared with the desired memories [e.g. <filename>.1.1.png, ..., <filename>.S.T.png, where S is the sample and T is the timestep]",)

    args, leftovers = ap.parse_known_args()

    with open(args.file) as f:
        test = json.load(f)
        test_args = test["arguments"]

    NRam(NRamContext(
        batch_size=args.batch_size if args.batch_size is not None else 2,
        max_int=args.max_int if args.max_int is not None else test_args["max_int"],
        timesteps=args.timesteps if args.timesteps is not None else test_args["time_steps"],
        task_type="task_%s" % test_args["task"],
        network=test["network"],
        gates=[ GateFactory.create(g) for g in test_args["gates"] ],
        debug_is_active=args.debug,
        print_circuits=args.print_circuits,
        print_memories=args.print_memories,
        path_config_file=os.path.abspath(args.file)
    )).execute()
