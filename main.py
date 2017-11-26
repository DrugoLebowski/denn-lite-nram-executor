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
        batch_size=2,
        max_int=10,
        num_regs=2,
        timesteps=3,
        task_type="task_access",
        network=network,
        num_hidden_layers=0,
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

    nram.execute()
