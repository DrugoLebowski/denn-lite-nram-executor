NRAM Executor
=============

Special thanks to [Andrew Gibiansky](http://andrew.gibiansky.com) for
[code inspirations](https://github.com/gibiansky/experiments) and [Gabriele Di Bari](https://github.com/Gabriele91) for his support.

This implementation of Neural Random-Access Machines ([https://arxiv.org/pdf/1511.06392.pdf](https://arxiv.org/pdf/1511.06392.pdf)) does not contains any learning algorithm, so don't use it for this scope.
Its only purpose is to test a pre-instructed neural network generated with [DENN-LITE](https://github.com/Gabriele91/DENN-LITE) over a specific task.

Anyway almost all the NRAM aspects are implemented, the only things missing out are cost calculation (that it is not useful for this scope) and some plotting functions.

Prerequisites
-------------
- Python 3.5+
- NumPy 1.14.0
- matplotlib 2.1
- pygraphviz 1.3.1
- tqdm 4.19.6
#### Install requirements 
```sh
$ pip install -r requirements.txt
```

Usage
-----
To test a configuration:

    $ python main.py tests/test_copy_working_best.json

To recall the help:

    $ python main.py -h

Flags
-----
#### --info
Activate the printing to console of every execution timestep, with all
the information about the gates and registers (like coefficients and values).

```sh
$ python main.py tests/example.json --info

OR

$ python main.py tests/example.json -i
```

#### --timesteps N+
The list of timesteps for which the NRAM should been run.
```sh
$ python main.py tests/example.py --timesteps 10 [...]

OR

$ python main.py tests/example.py -t 10 [...]
```

#### --max_int N+
The list of difficulties of integers for which the NRAM should work on.
```sh
$ python main.py tests/example.py --max_int 10 [...]

OR

$ python main.py tests/example.py -mi 10 [...]
```

#### --print_circuits [1 | 2]
With **1** activate the complete printing of the circuits and with **2** activate the pruned printing of the circuits.
```sh
$ python main.py tests/example.py --print_circuits 2

OR

$ python main.py tests/example.py -pc 2
```

#### --print_memories
Activate the printing of the memories status in TeX format.
```sh
$ python main.py tests/example.py --print_circuits 2

OR

$ python main.py tests/example.py -pc 2
```

#### --process_pool [1|2|3|4|...]
The number of process to spawn when the NRAM compute the samples.
```sh
$ python main.py tests/example.py --process_pool 8

OR

$ python main.py tests/example.py -pp 8
```
