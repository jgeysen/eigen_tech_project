import contextlib
import io
import sys


@contextlib.contextmanager
def no_stdout():
    """Yields a context in which one can run a (class) method where nothing is
    returned."""
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout
