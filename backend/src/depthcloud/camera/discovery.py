import multiprocessing as mp
from queue import Empty

from .worker import probe_main


def discover_cameras(max_index: int, timeout: float, backend="auto"):
    devices = []
    timed_out = []
    probe_errors = []
    ctx = mp.get_context("spawn")
    for index in range(max_index + 1):
        output = ctx.Queue(maxsize=1)
        process = ctx.Process(target=probe_main, args=(index, backend, output), daemon=True)
        process.start()
        process.join(timeout)
        if process.is_alive():
            timed_out.append(index)
            process.terminate()
            process.join(1)
        elif process.exitcode != 0:
            probe_errors.append({"index": index, "exit_code": process.exitcode})
        else:
            try:
                devices.append(output.get(timeout=0.1))
            except Empty:
                pass
        output.close()
        process.close()
    return {"devices": devices, "timed_out_indices": timed_out, "scanned_through": max_index, "probe_errors": probe_errors}


