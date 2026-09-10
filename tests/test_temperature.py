import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import meshcore_vanity as vanity


class TemperatureTests(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{value}\n", encoding="ascii")

    def test_millidegrees_are_bounded_and_converted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temp"
            for raw, expected in (
                    ("67450", 67.45), ("-40000", -40.0),
                    ("200000", 200.0), ("200001", None),
                    ("-40001", None), ("nan", None), ("", None)):
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="ascii")
                    self.assertEqual(vanity._read_millidegrees(path), expected)

    def test_cpu_uses_package_sensor_and_ignores_unrelated_devices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hwmon = root / "hwmon"
            thermal = root / "thermal"
            self.write(hwmon / "hwmon0/name", "nvme")
            self.write(hwmon / "hwmon0/temp1_label", "Composite")
            self.write(hwmon / "hwmon0/temp1_input", 99000)
            self.write(hwmon / "hwmon1/name", "coretemp")
            self.write(hwmon / "hwmon1/temp1_label", "Package id 0")
            self.write(hwmon / "hwmon1/temp1_input", 63000)
            self.write(hwmon / "hwmon1/temp2_label", "Core 0")
            self.write(hwmon / "hwmon1/temp2_input", 78000)
            self.write(thermal / "thermal_zone0/type", "x86_pkg_temp")
            self.write(thermal / "thermal_zone0/temp", 61000)
            self.assertEqual(
                vanity.read_cpu_package_temperature(hwmon, thermal), 63.0,
            )

    def test_cpu_supports_multiple_packages_amd_and_thermal_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hwmon = root / "hwmon"
            thermal = root / "thermal"
            self.write(hwmon / "hwmon0/name", "coretemp")
            self.write(hwmon / "hwmon0/temp1_label", "Package id 0")
            self.write(hwmon / "hwmon0/temp1_input", 61000)
            self.write(hwmon / "hwmon0/temp2_label", "Package id 1")
            self.write(hwmon / "hwmon0/temp2_input", 72000)
            self.assertEqual(
                vanity.read_cpu_package_temperature(hwmon, thermal), 72.0,
            )

            (hwmon / "hwmon0/name").write_text("k10temp\n", encoding="ascii")
            (hwmon / "hwmon0/temp1_label").write_text("Tctl\n", encoding="ascii")
            (hwmon / "hwmon0/temp1_input").write_text("68000\n", encoding="ascii")
            (hwmon / "hwmon0/temp2_label").write_text("Tccd1\n", encoding="ascii")
            (hwmon / "hwmon0/temp2_input").write_text("89000\n", encoding="ascii")
            self.assertEqual(
                vanity.read_cpu_package_temperature(hwmon, thermal), 68.0,
            )

            (hwmon / "hwmon0/name").write_text("nvme\n", encoding="ascii")
            self.write(thermal / "thermal_zone0/type", "x86_pkg_temp")
            self.write(thermal / "thermal_zone0/temp", 59000)
            self.assertEqual(
                vanity.read_cpu_package_temperature(hwmon, thermal), 59.0,
            )

    def test_cpu_missing_and_malformed_sensors_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root / "hwmon/hwmon0/name", "coretemp")
            self.write(root / "hwmon/hwmon0/temp1_label", "Package id 0")
            self.write(root / "hwmon/hwmon0/temp1_input", "not-a-number")
            self.write(root / "thermal/thermal_zone0/type", "acpitz")
            self.write(root / "thermal/thermal_zone0/temp", 75000)
            self.assertIsNone(vanity.read_cpu_package_temperature(
                root / "hwmon", root / "thermal",
            ))

    def test_pci_bus_ids_are_normalized(self) -> None:
        self.assertEqual(
            vanity.normalize_pci_bus_id("00000000:01:00.0"), "0000:01:00.0",
        )
        self.assertEqual(
            vanity.normalize_pci_bus_id("ABCD:0A:1F.7"), "abcd:0a:1f.7",
        )
        for invalid in (None, "01:00.0", "00010000:01:00.0", "0000:01:00.8"):
            with self.subTest(invalid=invalid):
                self.assertIsNone(vanity.normalize_pci_bus_id(invalid))

    def test_nvidia_parser_handles_reordering_and_unavailable_sensor(self) -> None:
        report = vanity.parse_nvidia_temperatures(
            "2, 00000000:65:00.0, 71\n"
            "0, 00000000:01:00.0, [N/A]\n"
            "1, 0000:17:00.0, 54\n"
        )
        self.assertEqual([reading.index for reading in report.readings], [0, 1, 2])
        self.assertEqual(report.readings[0].celsius, None)
        self.assertEqual(report.readings[1].pci_bus_id, "0000:17:00.0")
        self.assertEqual(report.readings[2].celsius, 71.0)

    def test_nvidia_parser_rejects_ambiguous_or_malformed_output(self) -> None:
        malformed = (
            "not csv",
            "0, not-a-bus, 50",
            "0, 0000:01:00.0, hot",
            "0, 0000:01:00.0, 201",
            "0, 0000:01:00.0, 50\n0, 0000:02:00.0, 51",
            "0, 0000:01:00.0, 50\n1, 00000000:01:00.0, 51",
        )
        for output in malformed:
            with self.subTest(output=output), self.assertRaises(ValueError):
                vanity.parse_nvidia_temperatures(output)

    def test_nvidia_query_is_bounded_and_optional(self) -> None:
        output = "0, 00000000:01:00.0, 47\n"
        with mock.patch.object(
                vanity, "_bounded_command_output", return_value=output,
        ) as bounded:
            report = vanity.query_nvidia_temperatures(timeout=0.75)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.readings[0].celsius, 47.0)
        command = bounded.call_args.args[0]
        self.assertIn("pci.bus_id", command[1])
        self.assertEqual(bounded.call_args.args[1], 0.75)
        self.assertEqual(
            bounded.call_args.args[2], vanity.TEMPERATURE_MAX_OUTPUT_BYTES,
        )
        with mock.patch.object(
                vanity, "_bounded_command_output", return_value=None,
        ):
            self.assertIsNone(vanity.query_nvidia_temperatures())

    def test_bounded_command_output_enforces_size_status_and_timeout(self) -> None:
        okay = vanity._bounded_command_output(
            [sys.executable, "-c", "print('okay')"], 1.0, 32,
        )
        self.assertEqual(okay, "okay\n")
        overflow = vanity._bounded_command_output(
            [sys.executable, "-c", "print('x' * 10000)"], 1.0, 128,
        )
        self.assertIsNone(overflow)
        failed = vanity._bounded_command_output(
            [sys.executable, "-c", "raise SystemExit(2)"], 1.0, 32,
        )
        self.assertIsNone(failed)
        started = time.monotonic()
        timed_out = vanity._bounded_command_output(
            [sys.executable, "-c", "import time; time.sleep(5)"], 0.05, 32,
        )
        self.assertIsNone(timed_out)
        self.assertLess(time.monotonic() - started, 1.0)

        # A faulty wrapper may exit after spawning a descendant that inherits
        # stdout. The inherited pipe must not extend our telemetry deadline.
        started = time.monotonic()
        inherited_pipe = vanity._bounded_command_output(
            [
                sys.executable,
                "-c",
                "import subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(1)'])",
            ],
            0.05,
            32,
        )
        self.assertIsNone(inherited_pipe)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_temperature_cache_stales_and_maps_by_pci_bus(self) -> None:
        cache = vanity.TemperatureCache()
        report = vanity.NvidiaTemperatureReport((
            vanity.NvidiaTemperatureReading(0, "0000:01:00.0", 46.0),
            vanity.NvidiaTemperatureReading(1, "0000:65:00.0", 72.0),
        ))
        vanity.update_temperature_cache(
            cache, vanity.TemperatureSnapshot(100.0, 64.0, report),
            stale_after=10.0,
        )
        self.assertEqual(
            vanity.format_temperature_status(
                cache, 1, "00000000:65:00.0", cuda_device_count=2,
            ),
            "CPU package 64 °C  •  GPU 1 72 °C",
        )
        self.assertEqual(
            vanity.format_temperature_status(
                cache, 0, None, cuda_device_count=2,
            ),
            "CPU package 64 °C  •  GPU 0 —",
        )

        vanity.update_temperature_cache(
            cache, vanity.TemperatureSnapshot(109.0, None, None),
            stale_after=10.0,
        )
        self.assertEqual(cache.cpu_celsius, 64.0)
        vanity.update_temperature_cache(
            cache, vanity.TemperatureSnapshot(110.0, None, None),
            stale_after=10.0,
        )
        self.assertIsNone(cache.cpu_celsius)
        self.assertEqual(cache.gpu_celsius_by_bus, {})

    def test_single_gpu_fallback_is_used_only_when_unambiguous(self) -> None:
        cache = vanity.TemperatureCache(
            gpu_celsius_by_bus={"0000:01:00.0": 55.0},
            gpu_observed_at_by_bus={"0000:01:00.0": 1.0},
            nvidia_device_count=1,
            nvidia_bus_ids=frozenset(("0000:01:00.0",)),
            nvidia_observed_at=1.0,
        )
        self.assertIn(
            "GPU 0 55 °C",
            vanity.format_temperature_status(cache, 0, None, 1),
        )
        self.assertIn(
            "GPU 0 —",
            vanity.format_temperature_status(cache, 0, None, 2),
        )

        # A successful replacement-device report with no temperature must not
        # borrow the previous physical GPU's still-within-grace reading.
        replacement = vanity.NvidiaTemperatureReport((
            vanity.NvidiaTemperatureReading(0, "0000:02:00.0", None),
        ))
        vanity.update_temperature_cache(
            cache, vanity.TemperatureSnapshot(2.0, None, replacement),
            stale_after=10.0,
        )
        self.assertIn(
            "GPU 0 —",
            vanity.format_temperature_status(cache, 0, None, 1),
        )

    def test_monitor_is_nonblocking_and_stops_without_late_callback(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        stop = threading.Event()
        delivered: list[vanity.TemperatureSnapshot] = []

        def blocked_sample() -> vanity.TemperatureSnapshot:
            entered.set()
            release.wait(2.0)
            return vanity.TemperatureSnapshot(time.monotonic(), 50.0, None)

        started = time.monotonic()
        worker = vanity.start_temperature_monitor(
            delivered.append, stop, interval=0.05, sampler=blocked_sample,
        )
        self.assertLess(time.monotonic() - started, 0.25)
        self.assertTrue(entered.wait(1.0))
        stop.set()
        release.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(delivered, [])

    def test_monitor_callback_runs_off_thread_and_wait_is_interruptible(self) -> None:
        caller = threading.get_ident()
        stop = threading.Event()
        delivered = threading.Event()
        callback_threads: list[int] = []

        def callback(_snapshot: vanity.TemperatureSnapshot) -> None:
            callback_threads.append(threading.get_ident())
            delivered.set()

        worker = vanity.start_temperature_monitor(
            callback, stop, interval=30.0,
            sampler=lambda: vanity.TemperatureSnapshot(
                time.monotonic(), 51.0, None,
            ),
        )
        self.assertTrue(delivered.wait(1.0))
        stop.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertNotEqual(callback_threads, [caller])


if __name__ == "__main__":
    unittest.main()
