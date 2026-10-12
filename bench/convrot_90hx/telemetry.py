"""Read-only NVML snapshots; sampled peaks are not allocator peaks."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time


class Telemetry:
    def __init__(self, path, uuid, interval=1.0):
        self.path, self.uuid, self.interval = path, uuid, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _run(self):
        fields = "timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate"
        with self.path.open("w") as stream:
            while not self.stop_event.is_set():
                row = {
                    "wall_time": time.time(),
                    "pid": os.getpid(),
                    "interval_sec": self.interval,
                }
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "-i",
                            self.uuid,
                            "--query-gpu=" + fields,
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=3,
                    )
                    row["gpu"] = dict(
                        zip(
                            fields.split(","),
                            [v.strip() for v in result.stdout.strip().split(",")],
                        )
                    )
                    processes = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=gpu_uuid,pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=3,
                    )
                    row["process_memory_mib"] = None
                    for line in processes.stdout.splitlines():
                        values = [v.strip() for v in line.split(",")]
                        if (
                            len(values) == 3
                            and values[0] == self.uuid
                            and values[1] == str(os.getpid())
                        ):
                            row["process_memory_mib"] = float(values[2])
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    row["error"] = str(exc)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                self.stop_event.wait(self.interval)
