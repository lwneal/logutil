#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import datetime
import json

from tqdm import tqdm
import numpy as np
import pytz
import tensorboard_logger

run_id = 'runs/run-{}'.format(int(time.time()))
tensorboard_logger.configure(run_id)

MAX_ENTRIES = 1000

# Input: Single-dimensional array x of size larger than desired_length
# Output: Array of length desired_length
def downsample(x, desired_length):
    indices = [int(np.round(i * len(x) / desired_length))
               for i in range(desired_length + 1)]
    return [np.mean(x[indices[i]:indices[i+1]]) for i in range(desired_length)]


def sparkline(data, length=16):
    BARS = u'▁▂▃▅▆▇'
    if len(data) > length:
        samples = downsample(data, length)
    else:
        step = len(data) / length
        samples = [data[int(step * i / length)] for i in range(length)]
    incr = min(samples)
    width = (max(samples) - min(samples)) / (len(BARS) - 1)
    bins = [i*width+incr for i in range(len(BARS))]
    indexes = [i for n in samples
                       for i, thres in enumerate(bins)
                                  if thres <= n < thres+width]
    return ''.join(BARS[i] for i in indexes)[:length]


# We assume x is a scalar.
# If x is not a scalar, that is a problem that we will fix right now.
def convert_to_scalar(x):
    if type(x).__name__ == 'FloatTensor':
        x = x.cpu()[0]
    elif type(x).__name__ == 'Variable':
        x = x.data.cpu()[0]
    try:
        return float(x)
    except:
        pass
    print('Warning: TimeSeries failed to convert scalar')
    return 0


def shorten(words, maxlen=30):
    if len(words) > maxlen - 3:
        words = words[:maxlen - 10] + '...' + words[-7:]
    return words


def formatted_time(timezone='America/Los_Angeles'):
    tz = pytz.timezone(timezone)
    fmt = '%I:%M:%S %p, %a %b %d %Y'
    now = datetime.datetime.now(tz).strftime(fmt)
    return now


class TimeSeries:
    def __str__(self):
        return self.format_all()

    def __init__(self, title=None, epoch_length=None, tensorboard=True):
        self.series = {}
        self.predictions = {}
        self.totals = {}
        self.start_time = time.time()
        self.last_printed_at = time.time()
        self.title = title
        self.epoch_length = epoch_length
        self.tensorboard = tensorboard

    def collect(self, name, value):
        if not self.series:
            self.start_time = time.time()
        if name not in self.series:
            self.series[name] = []
            self.totals[name] = 0
        if self.tensorboard:
            step = len(self.series[name])
            tensorboard_logger.log_value(name, value, step)
        self.series[name].append(convert_to_scalar(value))
        self.series[name] = self.series[name][-MAX_ENTRIES:]
        self.totals[name] += 1

    def collect_prediction(self, name, logits, ground_truth):
        if name not in self.predictions:
            self.predictions[name] = {'correct': 0, 'total': 0}
        _, pred_idx = logits.max(1)
        _, label_idx = ground_truth.max(1)
        correct = convert_to_scalar(sum(pred_idx == label_idx))
        self.predictions[name]['correct'] += correct
        self.predictions[name]['total'] += len(ground_truth)

    def print_every(self, n_sec=4):
        if time.time() - self.last_printed_at > n_sec:
            print(self.format_all())
            self.last_printed_at = time.time()

    def format_all(self):
        lines = ['']
        if self.title:
            lines.append(self.title)
        duration = time.time() - self.start_time
        lines.append("Collected {:.3f} sec ending {}".format(
            duration, formatted_time()))
        maxlen = max(self.totals.values())
        if self.epoch_length:
            lines.append(tqdm.format_meter(maxlen, self.epoch_length, duration, ncols=80))
        else:
            lines.append("Collected {:8d} points ({:.2f}/sec)".format(maxlen, maxlen / duration))
        lines.append("{:>32}{:>14}{:>16}".format('Name', 'Last {}'.format(MAX_ENTRIES), 'Last 10'))
        for name in sorted(self.series):
            values = np.array(self.series[name])
            name = shorten(name)
            lines.append("{:>32}:      {:8.4f}      {:8.4f} {}".format(
                name, values[-MAX_ENTRIES:].mean(), values[-10:].mean(), sparkline(values)))
        if self.predictions:
            lines.append('Predictions:')
        for name, pred in self.predictions.items():
            acc = 100 * pred['correct'] / pred['total']
            name = shorten(name)
            lines.append('{:>32}:\t{:.2f}% ({}/{})'.format(
                name, acc, pred['correct'], pred['total']))
        lines.append('\n')
        text = '\n'.join(lines)
        # Cache the most recent printed text to a file
        open('.last_summary.log', 'w').write(text)
        last_values = {key: self.series[key][-1] for key in self.series}
        open('.last_summary.json', 'w').write(json.dumps(last_values, indent=2))
        return text

    def write_to_file(self):
        filename = 'timeseries.{}.npy'.format(int(time.time()))
        ts = np.array(self.series[0])
        ts.save(filename)
        sys.stderr.write('Wrote array shape {} to file {}\n'.format(ts.shape, filename))
