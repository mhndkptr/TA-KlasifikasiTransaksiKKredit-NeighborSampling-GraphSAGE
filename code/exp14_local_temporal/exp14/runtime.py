"""Progress and persistent logs without changing sampling or model state."""
from contextlib import contextmanager
import logging
from pathlib import Path

from tqdm.auto import tqdm

from .legacy import EXP12_DIR  # Registers the shared EXP12 package.
from exp12.runtime import environment, progress


LOGGER = logging.getLogger('exp14')


class ProgressHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


@contextmanager
def logging_session(path):
    """Attach only our handlers; leave notebook/test/root logging untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [ProgressHandler(), logging.FileHandler(path, encoding='utf-8')]
    previous_level, previous_propagate = LOGGER.level, LOGGER.propagate
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    try:
        yield
    finally:
        for handler in handlers:
            LOGGER.removeHandler(handler)
            handler.close()
        LOGGER.setLevel(previous_level)
        LOGGER.propagate = previous_propagate


def progress_rows(groups, enabled, *, total, desc):
    """Count processed rows, including variable-sized user/merchant groups."""
    with progress(None, enabled, total=total, desc=desc, unit='txn', unit_scale=True) as bar:
        for group in groups:
            yield group
            bar.update(len(group))
