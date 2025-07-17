import logging
import json
import os
from datetime import datetime


class JSONFileHandler(logging.Handler):
    def __init__(self, filename):
        logging.Handler.__init__(self)
        self.filename = filename
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                json.dump([], f)  # Initialize with an empty list

    def emit(self, record):
        log_entry = self.format(record)
        with open(self.filename, "r+") as f:
            data = json.load(f)
            data.append(json.loads(log_entry))
            f.seek(0)
            json.dump(data, f, indent=4)


class CustomJSONFormatter(logging.Formatter):
    def __init__(self, **kwargs):
        self.extra_fields = kwargs
        fields = '", "'.join([f'{k}": "%({k})s' for k in kwargs.keys()])
        log_format = f'{{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "{fields}"}}'
        super().__init__(log_format)

    def format(self, record):
        for key, value in self.extra_fields.items():
            setattr(record, key, value)
        return super().format(record)


def setup_logger(name, filename, **kwargs):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = JSONFileHandler(filename)
    formatter = CustomJSONFormatter(**kwargs)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    # Example usage
    logger = setup_logger(
        "my_logger",
        r"C:\Projects\svo_tracker_experiments\RealTime\Performance Comparison\Velocity_estimation_results/log.json",
        commit_hash="838df96",
        dataset="116",
    )
    logger.info("This is an info message.")
    logger.error("This is an error message.")
