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
- Theano 1.0.1
- matplotlib 2.1
- pygraphviz 1.3.1
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
#### --debug
Activate the command line printing of every execution timestep, with all
the information about the gates and registers (like coefficients and values).

```sh
$ python main.py tests/example.json --debug

OR

$ python main.py tests/example.json -d
```

#### --timesteps N
Modify the running timesteps of NRAM.
```sh
$ python main.py tests/example.py --timesteps 10

OR

$ python main.py tests/example.py -t 10
```

#### --max_int N
The set of integers which the NRAM use.
```sh
$ python main.py tests/example.py --max_int 10

OR

$ python main.py tests/example.py -mi 10
```

#### --print_circuits
Activate the saving to an image file of the timestep's circuit for every sample.
```sh
$ python main.py tests/example.py --print_circuits

OR

$ python main.py tests/example.py -pc
```

#### --print_memories
Activate the saving to an image file of the memory status at the end of computation.
```sh
$ python main.py tests/example.py --print_memories

OR

$ python main.py tests/example.py -pm
```
