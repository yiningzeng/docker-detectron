# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Utilities for logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
from email.mime.text import MIMEText
import json
import logging
import numpy as np
import smtplib
import sys

def log_json_stats(stats, viz, win_accuracy_cls, win_loss, win_loss_bbox, win_loss_cls, sort_keys=True):
    # hack to control precision of top-level floats
    stats = {
        k: '{:.6f}'.format(v) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    print('json_stats: {:s}'.format(json.dumps(stats, sort_keys=sort_keys)))
    try:
        print("iter: " + str(stats.get("iter")))
        print("loss: " + str(stats.get("loss")))
        print("np iter: " + str(np.array([float(stats.get("iter"))])))
        print("np loss: " + str(np.array([float(stats.get("loss"))])))
        viz.line(
            X=np.array([float(stats.get("iter"))]),
            Y=np.array([float(stats.get("accuracy_cls"))]),
            win=win_accuracy_cls,
            update='append')
        viz.line(
            X=np.array([float(stats.get("iter"))]),
            Y=np.array([float(stats.get("loss"))]),
            win=win_loss,
            update='append')
        viz.line(
            X=np.array([float(stats.get("iter"))]),
            Y=np.array([float(stats.get("loss_bbox"))]),
            win=win_loss_bbox,
            update='append')
        viz.line(
            X=np.array([float(stats.get("iter"))]),
            Y=np.array([float(stats.get("loss_cls"))]),
            win=win_loss_cls,
            update='append')
    except:
        print('error: detectron/utils/logging.py log_json_stats')


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def AddValue(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def GetMedianValue(self):
        return np.median(self.deque)

    def GetAverageValue(self):
        return np.mean(self.deque)

    def GetGlobalAverageValue(self):
        return self.total / self.count


def send_email(subject, body, to):
    s = smtplib.SMTP('localhost')
    mime = MIMEText(body)
    mime['Subject'] = subject
    mime['To'] = to
    s.sendmail('detectron', to, mime.as_string())


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger
