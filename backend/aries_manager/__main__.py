"""Run the manager: python -m aries_manager [--host 0.0.0.0] [--port 8000]"""
import argparse
import logging

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Aries Image Studio manager")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port")
    parser.add_argument("--log-level", default="info", help="Python log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run("aries_manager.api:app", host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
