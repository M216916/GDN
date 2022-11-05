from util.data import get_attack_interval
import time
from datetime import datetime
from pytz import utc, timezone
from util.time import timestamp2str
import json
import argparse
import numpy as np

def printsep():
    print('='*40+'\n')
