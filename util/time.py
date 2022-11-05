import time
import math
from datetime import datetime
from pytz import utc, timezone


def timestamp2str(sec, fmt, tz):
    return datetime.fromtimestamp(sec).astimezone(tz).strftime(fmt)
