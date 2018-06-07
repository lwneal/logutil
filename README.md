# logutil

A simple library for logging statistics (eg. training loss) over time in
long-running Python scripts (eg. training PyTorch/Tensorflow models).

## Usage

Create a `TimeSeries` object. In your training loop, call `.collect()`
to gather statistics.

````
from logutil import TimeSeries
import numpy as np

ts = TimeSeries('My Metrics')
for i in range(1000):
    ts.collect('foo', i + np.random.normal())
    ts.collect('bar', np.sin(i * .01))
````


