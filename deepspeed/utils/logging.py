import logging
import sys
import os
import math

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


# Helper function to pretty-print message sizes
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# Helper function to calculate algbw and busbw.
# See https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36 and https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
def calc_bw(msg_size, lat):
    import deepspeed.comm as dist
    n = dist.get_world_size()
    algbw = ((msg_size * 8 * 2) / lat) / 1e6
    busbw = algbw * ((n - 1) / n)
    return algbw, busbw


class CommsLogger:
    def __init__(self, verbose=False):
        self.comms_dict = {}
        self.verbose = verbose

    # Add log entry
    def append(self, record_name, latency, msg_size):
        import deepspeed.comm as dist
        algbw, busbw = calc_bw(msg_size, latency)
        if record_name in self.comms_dict.keys():
            # If this comm_op has already been logged with this message size, just add to existing record
            if msg_size in self.comms_dict[record_name].keys():
                self.comms_dict[record_name][msg_size][0] += 1
                self.comms_dict[record_name][msg_size][1].append(latency)
                self.comms_dict[record_name][msg_size][2].append(algbw)
                self.comms_dict[record_name][msg_size][3].append(busbw)
            # If this is a new message size for this comm_op, add new record under existing comm_op
            else:
                self.comms_dict[record_name][msg_size] = [1, [latency], [algbw], [busbw]]
        else:
            # Create entirely new record
            self.comms_dict[record_name] = {msg_size: [1, [latency], [algbw], [busbw]]}
        # If verbose, print every comm op
        # TODO: Add to tensorboard
        if self.verbose:
            n = dist.get_world_size()
            log_str = f"rank={dist.get_rank()} | comm op: " + record_name + " | time (ms): {:.2f}".format(
                latency)
            log_str += " | msg size: " + convert_size(msg_size)
            log_str += " | algbw (Gbps): {:.2f} ".format(algbw)
            log_str += " | busbw (Gbps): {:.2f} ".format(busbw)
            log_dist(log_str, [0])

    # Print summary at end of iteration, epoch, or training
    def log_all(self):
        from deepspeed.utils.timer import trim_mean
        print(
            f"{'Comm. Op': <20}{'Message Size': <20}{'Count': <20}{'Total Latency(ms)': <20}{'Avg Latency(ms)': <20}{'tput_avg (Gbps)': <20}{'busbw_avg (Gbps)': <20}"
        )
        for record_name in self.comms_dict.keys():
            print(record_name)
            for msg_size, vals in sorted(self.comms_dict[record_name].items()):
                # vals[0] is the count for each msg size
                count = vals[0]
                # vals[1] is a list of latency records for each msg size
                total_lat = sum(vals[1])
                # vals[2] and vals[3] are the lists of algbw and busbw, respectively
                # Get rid of outliers when we print
                avg_lat = trim_mean(vals[1], 0.1)
                avg_algbw = trim_mean(vals[2], 0.1)
                avg_busbw = trim_mean(vals[3], 0.1)
                print(
                    f"{' ': <20}{convert_size(msg_size): <20}{count: <20}{total_lat: <20.2f}{avg_lat: <20.2f}{avg_algbw: <20.2f}{avg_busbw: <20.2f}"
                )


logger = LoggerFactory.create_logger(name="DeepSpeed", level=logging.INFO)


def log_dist(message, ranks=None, level=logging.INFO):
    import deepspeed.comm as dist
    """Log message when one of following condition meets

    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
        message (str)
        ranks (list)
        level (int)

    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        final_message = "[Rank {}] {}".format(my_rank, message)
        logger.log(level, final_message)


def print_json_dist(message, ranks=None, path=None):
    import deepspeed.comm as dist
    """Print message when one of following condition meets

    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
        message (str)
        ranks (list)
        path (str)

    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        message['rank'] = my_rank
        import json
        with open(path, 'w') as outfile:
            json.dump(message, outfile)
            os.fsync(outfile)


def get_current_level():
    """
    Return logger's current log level
    """
    return logger.getEffectiveLevel()


def should_log_le(max_log_level_str):
    """
    Args:
        max_log_level_str: maximum log level as a string

    Returns ``True`` if the current log_level is less or equal to the specified log level. Otherwise ``False``.

    Example:

        ``should_log_le("info")`` will return ``True`` if the current log level is either ``logging.INFO`` or ``logging.DEBUG``
    """

    if not isinstance(max_log_level_str, str):
        raise ValueError(f"{max_log_level_str} is not a string")

    max_log_level_str = max_log_level_str.lower()
    if max_log_level_str not in log_levels:
        raise ValueError(f"{max_log_level_str} is not one of the `logging` levels")

    return get_current_level() <= log_levels[max_log_level_str]
