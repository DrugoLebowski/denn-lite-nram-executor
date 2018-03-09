# Standard
import os
import shutil

# Vendor
import numpy as np

from tqdm import tqdm

# Project
from App import App
from NRamContext import NRamContext
from DebugTimestep import DebugTimestep
from activation_functions import relu, sigmoid, softmax
from util import print_memories, create_dir


class NRam(object):

    def __init__(self, context: NRamContext) -> None:
        self.context = context

    def execute(self) -> None:
        print("• Execution started")
        # Create the base directory for a task (e.g. TaskAccess)
        task_base_path = "%s%s" % (App.get("images_path"), self.context.tasks[0].__str__())
        create_dir(task_base_path)

        # Create the base directory for a test of DENN
        config_filename_without_extension = os.path.splitext(os.path.basename(self.context.path_config_file))[0]
        test_task_base_path = "%s/%s" % (task_base_path, config_filename_without_extension)
        create_dir(test_task_base_path, True) # Destroy all if the directory already exists

        # Copy the config file, for coherence
        shutil.copyfile(self.context.path_config_file, "%s/config.json" % test_task_base_path)

        for test_idx, task in enumerate(self.context.tasks):
            self.__test(test_idx, test_task_base_path, *task())
        print("• Execution terminated")

    def __test(self, test_idx, base_path, in_mem, out_mem, cost_mask, regs, timesteps) -> None:
        # Create a directory for the difficulty
        difficulty_test_base_path = "%s/%s" % (base_path, test_idx)
        create_dir(difficulty_test_base_path, True)

        # Iterate over sample
        for s in tqdm(range(self.context.batch_size)) if not self.context.info_is_active else range(self.context.batch_size):
            if self.context.info_is_active:
                print("\nSample[%d], Initial memory: %s, Desired memory: %s, Initial registers: %s"
                    % (s, in_mem[s, :].argmax(axis=1), out_mem[s], regs[s, :].argmax(axis=1)))

            # Iterate for every timestep
            self.context.debug.append(list()) # Init debug dictionary for the sample
            for t in range(timesteps):
                coeffs, _ = self.__run_network(regs[s])

                dt = DebugTimestep(self.context, t, s)
                regs[s], in_mem[s] = self.__run_circuit(regs[s], in_mem[s], self.context.gates, coeffs, dt)
                self.context.debug[s].append(dt)

            # Debug for the sample
            if self.context.info_is_active:
                for dt in self.context.debug[s]:
                    print(dt)
                print("\t• Expected mem => %s" % out_mem[s])

            if self.context.print_circuits:
                # Create dir for the single example of a difficulty
                sample_difficulty_base_path = "%s/%s" % (difficulty_test_base_path, s)
                create_dir(sample_difficulty_base_path, True)
                for dt in self.context.debug[s]:
                    dt.print_circuit(sample_difficulty_base_path)

        print_memories(self.context, in_mem, out_mem, cost_mask, difficulty_test_base_path, test_idx)


    def __avg(self, regs: np.array, coeff: np.array) -> np.array:
        """ Make the product between (registers + output of the gates)
            and a coefficient for the value selection """
        return np.array(
            np.tensordot(
                regs.transpose([1, 0]), coeff.transpose([1, 0]), axes=1
            ).transpose([1, 0]),
            dtype=np.float64)

    def __run_gate(self, gate_inputs, mem, gate, controller_coefficients):
        """Return the output of a gate in the circuit.

        gate_inputs:
          The values of the registers and previous gate outputs.
        gate:
          The gate to compute output for. Arity must
          match len(controller_coefficients).
        controller_coeffficients:
          A list of coefficient arrays from the controller,
          one coefficient for every gate input (0 for constants).
        """
        args = [self.__avg(gate_inputs, coefficients)
                for coefficients in controller_coefficients]
        mem, output = gate(mem, *args)

        # Special-case constant gates.
        # Since they have no outputs, they always output
        # one sample. Repeat their outputs as many times
        # as necessary, effectively doing manual broadcasting
        # to generate an output of the right size.
        if gate.arity == 0:
            output = output[None, ...]

        return output, mem, args

    def __run_circuit(self, registers: np.array, mem: np.array, gates: np.array,
                    controller_coefficients: np.array, debug: DebugTimestep) -> (np.ndarray, np.ndarray):
        # Initially, only the registers may be used as inputs.
        gate_inputs = registers

        # Debug purpose, dictionary for gates and regs history
        debug_step_gates = dict()
        debug_step_regs  = dict()

        # Run through all the gates.
        for i, (gate, coeffs) in enumerate(zip(gates, controller_coefficients)):
            output, mem, args = self.__run_gate(gate_inputs, mem, gate, coeffs)

            gate_info = dict()
            for i in range(gate.arity):
                gate_info[str(i)] = [coeffs[i].argmax(), args[i].argmax()]
            gate_info["res"] = output.argmax()
            debug_step_gates[gate.__str__()] = gate_info

            # Append the output of the gate as an input for future gates.
            gate_inputs = np.concatenate([gate_inputs, output])
        debug.gates = debug_step_gates
        debug.mem = mem.argmax(axis=1)

        # All leftover coefficients are for registers.
        for i, coeff in enumerate(controller_coefficients[len(gates):]):
            gate_inputs[i] = self.__avg(gate_inputs, coeff)
            debug_step_regs[str(i)] = [coeff.argmax(), gate_inputs[i].argmax()]
        debug.regs = debug_step_regs

        return gate_inputs[np.arange(self.context.num_regs)], mem

    def __run_network(self, registers: np.array) -> np.array:

        def take_params(values, i):
            """Return the next pair of weights and biases after the
            starting index and the new starting index."""
            return values[i], values[i + 1], i + 2

        # Extract the 0th (i.e. P( x = 0 )) component from all registers.
        last_layer = np.array(registers[:, 0][None, ...], dtype=np.float64)

        # Propogate forward to hidden layers.
        idx = 0
        for i in range(self.context.num_hidden_layers):
            W, b, idx = take_params(self.context.network, idx)
            last_layer = relu(last_layer.dot(W) + b)

        controller_coefficients = []
        for i, gate in enumerate(self.context.gates):
            coeffs = []
            for j in range(gate.arity):
                W, b, idx = take_params(self.context.network, idx)
                layer = softmax(last_layer.dot(W) + b)
                coeffs.append(layer)
            controller_coefficients.append(coeffs)

        # Forward propogate to new register value coefficients.
        for i in range(self.context.num_regs):
            W, b, idx = take_params(self.context.network, idx)
            coeffs = softmax(last_layer.dot(W) + b)
            controller_coefficients.append(coeffs)

        # Forward propogate to generate willingness to complete.
        W, b, idx = take_params(self.context.network, idx)
        complete = sigmoid(last_layer.dot(W) + b)

        return controller_coefficients, complete
