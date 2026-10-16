import time

def retry_with_backoff(fn, max_retries=3, base_delay=1.0):
    last_err = None
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** i))
    raise last_err
