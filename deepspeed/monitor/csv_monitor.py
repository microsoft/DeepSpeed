from .monitor import Monitor


class csvMonitor(Monitor):
    def __init__(self, monitor_config):
        super().__init__(monitor_config)
        import csv
        self.filenames = []
        self.csv_monitor_output_path = monitor_config.csv_monitor.output_path

    def write_events(self, event_list):
        import csv
        # We assume each event_list element is a tensorboard-style tuple in the format: (log_name: String, value, step: Int)
        for event in event_list:
            log_name = event[0]
            value = event[1]
            step = event[2]

            # Set the header to the log_name
            # Need this check because the deepspeed engine currently formats log strings to separate with '/'
            if '/' in log_name:
                record_splits = log_name.split('/')
                header = record_splits[len(record_splits) - 1]
            else:
                header = log_name

            # sanitize common naming conventions into filename
            filename = log_name.replace('/', '_').replace(' ', '_')
            fname = self.csv_monitor_output_path + filename + '.csv'

            # Open file and record event. Insert header if this is the first time writing
            with open(fname, 'a+') as csv_monitor_file:
                csv_monitor_writer = csv.writer(csv_monitor_file)
                if filename not in self.filenames:
                    self.filenames.append(filename)
                    csv_monitor_writer.writerow(['step', header])
                csv_monitor_writer.writerow([step, value])
