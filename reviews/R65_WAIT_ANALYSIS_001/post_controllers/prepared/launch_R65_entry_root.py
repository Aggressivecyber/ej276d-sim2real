"""Launch exactly one eligible post entry after root review; never train/recover."""
import json
from r65_controller_common import parser, run_owned, launch_entry


def main():
    args = parser(__doc__, with_entry=True).parse_args()
    result = run_owned(args, "launch", lambda owner: launch_entry(owner, args.entry))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
