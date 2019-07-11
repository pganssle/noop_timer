# NOP Timer

This is a script that calculates how much time it takes to run a function consisting of N NOP bytecode operations (these have no effect and are usually removed as part of the peephole optimization step), as a function of N.

## Installation

To use this, run `pip install noop_timer`. For convenience, it is probably best to use a virtual environment:

```
python -m virtualenv noop_venv --python=3.7
source noop_venv/bin/activate
pip install noop_timer
```

## Use
To run, execute the module with `python -m noop_timer`.
