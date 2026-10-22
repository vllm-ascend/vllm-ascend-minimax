# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Add the unique draft architecture to the known-DSpark allowlist only."""

import argparse
import difflib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("speculative_py", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    before = args.speculative_py.read_text(encoding="utf-8")
    needle = '                    and "K3DSparkModel" not in self.draft_model_config.architectures\n'
    addition = '                    and "MiniMaxM31FlashA3DSparkModel" not in self.draft_model_config.architectures\n'
    needle_count, addition_count = before.count(needle), before.count(addition)
    if needle_count != 1 or addition_count > 1 or (addition_count == 1 and needle + addition not in before):
        raise RuntimeError("Core version changed: review the DSpark guard before applying")
    if addition_count == 1:
        print("Guard already contains the MiniMax draft architecture")
    else:
        after = before.replace(needle, needle + addition)
        print(
            "".join(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile=str(args.speculative_py),
                    tofile=str(args.speculative_py),
                )
            )
        )
        if args.apply:
            backup = args.speculative_py.with_suffix(".py.before-minimax-dspark-flash")
            if backup.exists():
                raise FileExistsError(backup)
            backup.write_text(before, encoding="utf-8")
            args.speculative_py.write_text(after, encoding="utf-8")


if __name__ == "__main__":
    main()
