import os
import functools
import signal
from pathlib import Path

RESOURCE_ROOT = os.environ.get("RESOURCE_ROOT", "./client_resources")


def get_real_path(path):
    if path is None:
        return None
    if RESOURCE_ROOT in path:
        return path
    return os.path.join(RESOURCE_ROOT, path)


def get_root_dir():
    return RESOURCE_ROOT


def md2plain(md):
    plain_text = md.replace("&nbsp;", " ")
    plain_text = plain_text.replace("<br>", "\n")
    plain_text = plain_text.replace("\<", "<")
    plain_text = plain_text.replace("\>", ">")
    return plain_text


def plain2md(plain_text: str):
    md_text = plain_text.replace("<", "\<")
    md_text = md_text.replace(">", "\>")
    md_text = md_text.replace("\n", "<br>")
    # md_text = md_text + "<br>"
    md_text = md_text.replace(" ", "&nbsp;")
    return md_text


def transform_msgs(history_msgs: list = []):
    if history_msgs is None:
        return []
    filtered_msg = []
    for item in history_msgs:
        if isinstance(item[0], str):
            item[0] = md2plain(item[0])
        if isinstance(item[1], str):
            item[1] = md2plain(item[1])
        if isinstance(item[1], str) and item[1].startswith(
            "The whole process will take some time, please be patient."
        ):
            item[1] = None

        filtered_msg.append(item)
    return filtered_msg


def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            def _handle_timeout(signum, frame):
                err_msg = f"Function {func.__name__} timed out after {sec} seconds"
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func

    return decorator
