# Hardware validation

MeshCore CUDA Vanity Key Generator v1.0.0 was interoperability-tested on
2026-09-08 using a RAKwireless WisCore RAK4631 running MeshCore companion
firmware `v1.17.1-d929643`.

## Procedure

1. Export and securely back up the device's existing identity.
2. Derive its public key and complete an Ed25519 sign/verify check locally.
3. Generate a temporary identity with the optimized CUDA engine.
4. Independently validate the generated private/public pair on the CPU.
5. Import the expanded 64-byte private key into the RAK4631.
6. Query the device and confirm its public key changed to the expected value.
7. Export the device identity and compare it byte-for-byte with the generated
   private key.
8. Ask the RAK4631 to sign a test challenge and verify the returned signature
   against the generated public key.
9. Restore the original identity and confirm both its exported private key and
   reported public key byte-for-byte.

## Test identity and result

- Public key:
  `c0dec0596681db0273453ed24c9716bbea84006554ff5c9f1c6274a8cdcfa2a5`
- Requested prefix: `c0dec0`
- CUDA engine: `optimized`
- Import and exact private-key export comparison: **PASS**
- On-device signature verification: **PASS**
- Original identity restoration and export verification: **PASS**

No private keys are included in this document. No LoRa packet was transmitted
during the test; it validated identity import, storage, derivation, and signing
through the firmware's local companion interface.
