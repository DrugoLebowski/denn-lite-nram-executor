# Standard
import concurrent.futures
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
            difficulty_test_base_path = "%s/%s" % (test_task_base_path, test_idx)
            create_dir(difficulty_test_base_path, True)

            # Retrieve batch of difficulty
            in_mem, out_mem, cost_mask, regs, timesteps = task()
            # Iterate over sample
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.context.process_pool) as executor:
                futures = [executor.submit(self.execute_test_sample, self.context, s,difficulty_test_base_path,
                                           in_mem[s], out_mem[s], regs[s], timesteps)
                           for s in range(self.context.batch_size)]

                for f in tqdm(concurrent.futures.as_completed(futures), total=self.context.batch_size) \
                        if not self.context.info_is_active else concurrent.futures.as_completed(futures):
                    s, modified_in_mem = f.result()
                    in_mem[s] = modified_in_mem
                print_memories(in_mem, out_mem, cost_mask, difficulty_test_base_path, test_idx)
        print("• Execution terminated")

    def execute_test_sample(self, context, s, difficulty_test_base_path, in_mem, out_mem, regs, timesteps):
        if context.print_memories or context.print_circuits is not 0:
            sample_difficulty_base_path = "%s/%s" % (difficulty_test_base_path, s)
            create_dir(sample_difficulty_base_path, True)

        if context.info_is_active:
            print("\nSample[%d], Initial memory: %s, Desired memory: %s, Initial registers: %s"
                  % (s, in_mem[:].argmax(axis=1), out_mem, regs[:].argmax(axis=1)))

        # Iterate for every timestep
        debug = list()  # Init debug dictionary for the sample
        for t in range(timesteps):
            coeffs, _ = self.__run_network(regs)

            dt = DebugTimestep(context, t, s)
            regs, in_mem = self.__run_circuit(regs, in_mem, context.gates, coeffs, dt)
            debug.append(dt)

        # Debug for the sample
        if context.info_is_active:
            for dt in debug:
                print(dt)
            print("\t• Expected mem => %s" % out_mem)

        if context.print_memories:
            with open("%s/memories.txt" % sample_difficulty_base_path, "a+") as f:
                f.write("\\textbf{Step} & %s & %s & Read & Write \\\\ \hline \n"
                        % (" & ".join(["%s" % r for r in range(out_mem.shape[0])]),
                           " & ".join(["\\textit{r}%d" % r for r in range(context.num_regs)])))
            for dt in debug:
                dt.print_memory_to_file(sample_difficulty_base_path, timesteps)

        if context.print_circuits is not 0:
            # Create dir for the single example of a difficulty
            for dt in debug:
                if context.print_circuits is 1:
                    dt.print_circuit(sample_difficulty_base_path)
                else:
                    dt.print_pruned_circuit(sample_difficulty_base_path)

        return s, in_mem

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

        debug.mem_previous_mod = mem.argmax(axis=1)
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
