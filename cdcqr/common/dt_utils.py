import math
import time
from datetime import datetime
from time import mktime


class DataTimeUtils:
    def __init__(self):
        pass

    @staticmethod
    def epoch_time2ts_s(epoch_time):
        """
        Returns the parsed time compoents from epoch time upto milisecond

        Returns:
                dt (datetime) up to second
        """
        dt = epoch_time // 1e6
        struct_time = time.localtime(dt)
        dt1 = datetime.fromtimestamp(mktime(struct_time))
        return dt1

    @staticmethod
    def ceil_dt(delta, dt):
        """
        return the next nearest time point with delta granularity
        """
        ref_dt = datetime(1970, 1, 1)
        return ref_dt + math.ceil((dt - ref_dt) / delta) * delta
