import contextlib
import io
import sys


@contextlib.contextmanager
def no_stdout():
    """Yields a context in which one can run a (class) method where nothing is
    returned.

    Yields:
        context: methods called within this context will not return anything.
    """
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout
