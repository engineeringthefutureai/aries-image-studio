"""CLI entrypoint for aries_worker.
Spawned by the manager, one process per (stage, card):
  python -m aries_worker --stage <enhance|encode|denoise|decode> --card <n> --socket <path>
"""
import argparse
import asyncio
import logging
import sys

from aries_worker.client import WorkerSocketClient
from aries_worker.stages.enhance import EnhanceStageRunner
from aries_worker.stages.encode import EncodeStageRunner
from aries_worker.stages.denoise import DenoiseStageRunner
from aries_worker.stages.decode import DecodeStageRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] aries_worker: %(message)s",
)
logger = logging.getLogger("aries_worker")


STAGE_RUNNERS = {
    "enhance": EnhanceStageRunner,
    "encode": EncodeStageRunner,
    "denoise": DenoiseStageRunner,
    "decode": DecodeStageRunner,
}


def main():
    parser = argparse.ArgumentParser(description="Aries Image Studio Stage Worker")
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_RUNNERS))
    parser.add_argument("--card", required=True, type=int, help="Card index, as numbered by the manager")
    parser.add_argument("--socket", required=True, help="Path to manager Unix domain socket")
    parser.add_argument("--idle-grace", type=float, default=7.0, help="Seconds to wait while idle before exiting")
    args = parser.parse_args()

    logger.info("Initializing stage worker for '%s' on card %d...", args.stage, args.card)
    runner = STAGE_RUNNERS[args.stage]()
    runner.load()  # load before connecting: `hello` means "ready to claim"

    client = WorkerSocketClient(
        stage=args.stage,
        card=args.card,
        socket_path=args.socket,
        runner=runner,
        idle_grace=args.idle_grace,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user.")
    except Exception as e:
        logger.exception("Worker terminated with error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
