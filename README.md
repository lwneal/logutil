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

To view your statistics, just print the TimeSeries object:

````
>>> print(ts)
My Metrics
Collected 1.736 sec ending 12:52:47 PM, Thu Jun 07 2018
Collected     1000 points (576.04/sec)
                            Name          Avg.         Last 10
                             bar:        0.1842       -0.4969   ▅▆▆▇▆▅▂▁▁▁▂▅▆▆▆▅▃
                             foo:      499.5061      994.6164   ▁▁▁▁▂▂▃▃▃▅▅▅▅▆▆▇
````

By default, all values are also written to a Tensorboard log in the
`runs/` directory:

![screenshot](https://raw.githubusercontent.com/lwneal/logutil/master/screenshot.png)
