"""Read frozen R65 native/metadata gates; this command cannot launch anything."""
import json
from r65_controller_common import parser, run_owned, remote_call


def main():
    args = parser(__doc__).parse_args()
    result = run_owned(args, "poll", lambda owner: remote_call(owner, "observe"))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
