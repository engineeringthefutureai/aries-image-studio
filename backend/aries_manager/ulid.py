"""ULID generator (26 characters, Crockford Base32): 48-bit millisecond time + 80 random bits."""
import secrets
import time

CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(CROCKFORD_BASE32[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def generate_ulid() -> str:
    t_ms = int(time.time() * 1000)
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    return _encode(t_ms, 10) + _encode(rand, 16)
