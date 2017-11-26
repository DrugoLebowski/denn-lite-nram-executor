import numpy as np
import json

from Nram import NRam
from NRamContext import NRamContext
from GateFactory import GateFactory

if __name__ == "__main__":
    # TODO: read arguments from command line


    with open("test.json") as f:
        test = json.load(f)

    network = test["network"]

    context = NRamContext(
        batch_size=1,
        max_int=10,
        num_regs=4,
        timesteps=9,
        task_type="task_copy",
        network=network,
        num_layers=0,
        gates=[
            GateFactory.create("read"),
            GateFactory.create("zero"),
            GateFactory.create("inc"),
            GateFactory.create("lt"),
            GateFactory.create("min"),
            GateFactory.create("write"),
        ]
    )

    nram = NRam(context)

    print(nram.execute())
