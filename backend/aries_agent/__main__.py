"""CLI entrypoint for aries_agent (Architecture V3 §10).

Usage:
  python -m aries_agent
"""
import argparse
import asyncio
import logging
import sys

from aries_agent.agent import AriesAgent
from aries_agent.config import AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] aries_agent: %(message)s",
)
logger = logging.getLogger("aries_agent")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aries Image Studio Thin Agent (Architecture V3)")
    parser.add_argument("--agent-id", default=None, help="Agent identifier (overrides ARIES_AGENT_ID)")
    parser.add_argument("--grpc-url", default=None, help="Backend gRPC address (overrides ARIES_BACKEND_GRPC_URL)")
    parser.add_argument("--http-url", default=None, help="Backend HTTP origin (overrides ARIES_BACKEND_HTTP_URL)")
    parser.add_argument("--metrics-port", type=int, default=None, help="Agent /metrics port")
    args = parser.parse_args()

    cfg = AgentConfig.from_env()
    if args.agent_id:
        cfg.agent_id = args.agent_id
    if args.grpc_url:
        cfg.backend_grpc_url = args.grpc_url
    if args.http_url:
        cfg.backend_http_url = args.http_url.rstrip("/")
    if args.metrics_port is not None:
        cfg.metrics_port = args.metrics_port

    # §10.1 guard: The agent never imports torch
    if "torch" in sys.modules:
        raise RuntimeError("aries_agent must never import torch")

    agent = AriesAgent(config=cfg)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user.")


if __name__ == "__main__":
    main()
