import logging
import sys
import os
import math

#import deepspeed.comm as dist

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


def convert_size(size_bytes, is_bw=False):
    if size_bytes == 0:
        return "0B"
    if is_bw:
        size_name = ("B/s",
                     "KB/s",
                     "MB/s",
                     "GB/s",
                     "TB/s",
                     "PB/s",
                     "EB/s",
                     "ZB/s",
                     "YB/s")
    else:
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class CommsLogger:
    def __init__(self, verbose=False):
        self.comms_dict = {}
        self.verbose = verbose

    def append(self, record_name, latency, msg_size):
        import deepspeed.comm as dist
        if record_name in self.comms_dict.keys():
            if msg_size in self.comms_dict[record_name].keys():
                #print(self.comms_dict[record_name])
                self.comms_dict[record_name][msg_size][0] += 1
                self.comms_dict[record_name][msg_size][1] += latency
                #self.comms_dict[record_name][msg_size][2] += msg_size / latency
            else:
                #self.comms_dict[record_name][msg_size] = [1, latency, msg_size/latency]
                self.comms_dict[record_name][msg_size] = [1, latency]
        else:
            #self.comms_dict[record_name] = {msg_size: [1, latency, msg_size / latency]}
            self.comms_dict[record_name] = {msg_size: [1, latency]}
        if self.verbose:
            log_str = f"rank={dist.get_rank()} time (ms)" + " | {}: {:.2f}".format(
                record_name,
                latency)
            log_str += " | msg size " + convert_size(msg_size)
            log_str += " | BW " + convert_size(round(msg_size / latency, 2), is_bw=True)
            log_dist(log_str, [0])

    def log_all(self):
        print(self.comms_dict)
        for record_name in self.comms_dict.keys():
            #print(record_name + ":")
            print(
                "Message Size\t\t\t\tCount\t\t\t\tTotal Latency(us)\t\t\t\tAvg Latency(us)\t\t\t\tAvg BW"
            )
            for msg_size, vals in self.comms_dict[record_name].items():
                count = vals[0]
                total_lat = round(vals[1], 2)
                avg_lat = round(total_lat / count, 2)
                avg_bw = round((msg_size * count) / total_lat, 2)
                #entry_str += str(msg_size) + '\t' * (len(msg_size))
                #entry_str += "\t\t\t\t\t" + str(vals[0]) + "\t\t\t\t" + str(vals[1]) + "\t\t\t" + str(vals[2])
                print(
                    convert_size(msg_size) + "\t\t\t\t\t" + str(count) + "\t\t\t\t" +
                    str(total_lat) + "\t\t\t" + str(avg_lat) + "\t\t\t" +
                    convert_size(avg_bw,
                                 is_bw=True))


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
