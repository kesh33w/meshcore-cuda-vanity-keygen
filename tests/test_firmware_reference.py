"""Independent reference checks for MeshCore's Ed25519 key exchange.

The formulas mirror MeshCore ``lib/ed25519/key_exchange.c`` at upstream commit
``d92964352441e53b93e8667b802e04f6e072b39e``. They intentionally do not call
libsodium, so this catches a conversion or scalar-multiplication mismatch in the
production ctypes path.
"""

import unittest

import meshcore_vanity as vanity


FIELD_PRIME = 2**255 - 19
MONTGOMERY_A24 = 121665


def edwards_public_to_montgomery_u(public: bytes) -> int:
    """Apply MeshCore's u = (y + 1) / (1 - y) conversion."""
    encoded = int.from_bytes(public, "little")
    y = encoded & ((1 << 255) - 1)
    return ((y + 1) * pow(1 - y, FIELD_PRIME - 2, FIELD_PRIME)) % FIELD_PRIME


def meshcore_reference_exchange(private_scalar: bytes, public: bytes) -> bytes:
    """Small, non-constant-time test reference for MeshCore's ladder."""
    scalar = bytearray(private_scalar)
    scalar[0] &= 248
    scalar[31] &= 63
    scalar[31] |= 64
    integer = int.from_bytes(scalar, "little")
    u = edwards_public_to_montgomery_u(public)

    x_1 = u
    x_2, z_2 = 1, 0
    x_3, z_3 = u, 1
    swap = 0
    for position in range(254, -1, -1):
        bit = (integer >> position) & 1
        swap ^= bit
        if swap:
            x_2, x_3 = x_3, x_2
            z_2, z_3 = z_3, z_2
        swap = bit

        a = (x_2 + z_2) % FIELD_PRIME
        aa = a * a % FIELD_PRIME
        b = (x_2 - z_2) % FIELD_PRIME
        bb = b * b % FIELD_PRIME
        e = (aa - bb) % FIELD_PRIME
        c = (x_3 + z_3) % FIELD_PRIME
        d = (x_3 - z_3) % FIELD_PRIME
        da = d * a % FIELD_PRIME
        cb = c * b % FIELD_PRIME
        x_3 = (da + cb) ** 2 % FIELD_PRIME
        z_3 = x_1 * (da - cb) ** 2 % FIELD_PRIME
        x_2 = aa * bb % FIELD_PRIME
        z_2 = e * (aa + MONTGOMERY_A24 * e) % FIELD_PRIME

    if swap:
        x_2, x_3 = x_3, x_2
        z_2, z_3 = z_3, z_2
    shared = x_2 * pow(z_2, FIELD_PRIME - 2, FIELD_PRIME) % FIELD_PRIME
    return shared.to_bytes(32, "little")


class FirmwareReferenceTests(unittest.TestCase):
    def test_independent_reference_matches_meshcore_vector_and_production(self):
        public = bytes.fromhex(vanity.TEST_PUBLIC)
        forward = meshcore_reference_exchange(
            vanity.TEST_PRIVATE[:32], vanity.ECDH_TEST_PEER_PUBLIC
        )
        reverse = meshcore_reference_exchange(
            vanity.ECDH_TEST_PEER_PRIVATE[:32], public
        )

        self.assertEqual(forward, vanity.ECDH_TEST_SHARED)
        self.assertEqual(reverse, vanity.ECDH_TEST_SHARED)
        self.assertEqual(
            vanity.meshcore_shared_secret(
                vanity.TEST_PRIVATE, vanity.ECDH_TEST_PEER_PUBLIC
            ),
            forward,
        )


if __name__ == "__main__":
    unittest.main()
