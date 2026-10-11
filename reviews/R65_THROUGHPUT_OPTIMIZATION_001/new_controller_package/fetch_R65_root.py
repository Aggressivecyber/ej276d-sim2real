"""Fetch a truly completed R65 entry, verify bytes, preserve previous versions."""
import json
from r65_controller_common import parser, run_owned, fetch_entry


def main():
    args = parser(__doc__, with_entry=True, allow_train=True).parse_args()
    result = run_owned(args, "fetch", lambda owner: fetch_entry(owner, args.entry))
    print(json.dumps({key: value for key, value in result.items() if key != "actions"}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
