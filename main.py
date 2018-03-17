# Standard
import os
import json
from argparse import ArgumentParser

# Project
from App import App
from NRam import NRam
from NRamContext import NRamContext
from factories.GateFactory import GateFactory
from util import create_dir


if __name__ == "__main__":
    create_dir(App.get("images_path"))

    ap = ArgumentParser()

    ap.add_argument("file",
                    type=str,
                    help="The file where resides the configuration of NRAM", )
    ap.add_argument("--batch_size", "-bs",
                    dest="batch_size",
                    type=int,
                    default=2,
                    help="The examples to analyze", )
    ap.add_argument("--timesteps", "-t",
                    dest="timesteps",
                    nargs="+",
                    type=int,
                    required=True,
                    help="The timesteps that the NRAM must execute", )
    ap.add_argument("--max_ints", "-mi",
                    dest="max_ints",
                    nargs="+",
                    type=int,
                    required=True,
                    help="The dimensions of the sets of numbers", )
    ap.add_argument("--info", "-i",
                    dest="info",
                    nargs="?",
                    const=True,
                    default=False,
                    help="Write out the info of NRAM, like memories and registers", )
    ap.add_argument("--print_circuits", "-pc",
                    dest="print_circuits",
                    type=int,
                    default=0,
                    help="Draw the circuit of each timestep to a file with a progressive numeration [e.g. "
                         "<filename>.1.1.png, ..., <filename>.S.T.png, where S is the sample and T is the timestep]. "
                         "With: "
                         "  • 0 the circuits are not printed; "
                         "  • 1 the circuits are printed entirely; "
                         "  • 2 the circuits are printed pruning the Gates or Registers R* which have not a path to a Register(s) R'*.",)
    ap.add_argument("--print_memories_in_step_to_file", "-pmtf",
                    dest="print_memories",
                    nargs='?',
                    const=True,
                    default=False,
                    help="For each sample and for each step, write to file the memory and the registers.", )

    args, leftovers = ap.parse_known_args()

    with open(args.file) as f:
        test = json.load(f)
        test_args = test["arguments"]

    NRam(NRamContext(
        batch_size=args.batch_size,
        l_max_int=args.max_ints,
        l_timesteps=args.timesteps,
        task_type="task_%s" % test_args["task"],
        network=test["network"],
        gates=[ GateFactory.create(g) for g in test_args["gates"] ],
        info_is_active=args.info,
        print_circuits=args.print_circuits,
        print_memories=args.print_memories,
        path_config_file=os.path.abspath(args.file),
    )).execute()
