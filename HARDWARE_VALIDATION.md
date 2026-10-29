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

## v1.10.0 on Heltec V4.3 (2026-10-28)

The v1.10.0 optimized engine (affine walk with `fe32` arithmetic) was
re-validated on a Heltec V4.3 OLED (ESP32-S3, 16 MB flash, 2 MB PSRAM) freshly
flashed with the official MeshCore USB companion firmware `v1.18.0-81dbae5`
(`heltec_v4_companion_radio_usb-v1.18.0-81dbae5-merged.bin`, SHA-256
`d5b0f7a152755b9bc57b1dae1ed3728aea01893468ae9be1ccfaa02d0e4baafc`). The
procedure was the same as above, over the USB companion protocol, with a
reboot after each import.

- Public key:
  `c0de110ee7a299d2d39fabd2497a02072b0851260a9951f4e0bd998d2722edcd`
- Requested prefix: `c0de110`
- CUDA engine: `optimized` (v1.10.0)
- Device-reported public key after import: **PASS**
- Exact private-key export comparison: **PASS**
- On-device signature verification: **PASS**
- Original identity restoration and export verification: **PASS**

As before, no private keys are included and no LoRa packet was transmitted.

## v1.11.0 on Heltec V4.3 (2026-10-28)

The v1.11.0 engine computes only y and resolves the sign of x lazily. It was
validated on the same Heltec V4.3 and firmware as above, using a key whose
sign bit (bit 255) is set so the lazily resolved bit is exercised.

- Public key:
  `c0de113d2b7025e033324c00b2d2c28c3b7b9902efab82e0c59f9a4320456ff0`
- Requested pattern: prefix `c0de11`, suffix `f0`
- CUDA engine: `optimized` (v1.11.0)
- Device-reported public key after import: **PASS**
- Exact private-key export comparison: **PASS**
- On-device signature verification: **PASS**
- Original identity restoration and export verification: **PASS**
