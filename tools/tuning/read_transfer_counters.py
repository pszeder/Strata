#!/usr/bin/env python3
"""Read the diagnostic HIP copy/staging counters; these are not physical bus traffic."""
import argparse
import json
import struct
from pathlib import Path

MAGIC = 0x5354524154414354


def snapshot(path):
    data = path.read_bytes()
    if len(data) < 88:
        raise ValueError("counter file is truncated (need 88 bytes)")
    values = struct.unpack_from("<11Q", data)
    if values[0] != MAGIC or values[8] != 2:
        raise ValueError("expected Strata transfer monitor v2 counters")
    names = ("magic", "pid", "hip_h2d_bytes", "hip_d2h_bytes", "hip_d2d_bytes",
             "hip_other_bytes", "hip_copy_calls", "mapped_staging_bytes", "version",
             "mapped_staging_copy_calls", "mapped_range_overflow")
    return dict(zip(names, values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="private STRATA_TRANSFER_COUNTER_FILE")
    args = parser.parse_args()
    try:
        result = snapshot(args.file)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    result["scope"] = ("HIP API payload and large mapped staging writes; not physical PCIe/DRAM. "
                       "Individual fields are atomic; this snapshot is not transactional. "
                       "A nonzero mapped_range_overflow invalidates complete staging accounting.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
