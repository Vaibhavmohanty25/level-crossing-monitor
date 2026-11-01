import unittest

from crossing_monitor.escalation import EscalationMonitor


class EscalationTests(unittest.TestCase):
    def test_thresholds_and_slow_continuous_frames(self):
        monitor = EscalationMonitor()

        for timestamp, expected in [
            (0, "ENTRY"),
            (2.9, "ENTRY"),
            (3, "PERSISTENT"),
            (6, "PROLONGED"),
        ]:
            states = monitor.update({1}, {1}, timestamp, True)
            self.assertEqual(states[1].level, expected)

    def test_open_crossing_clears_and_reclosing_restarts(self):
        monitor = EscalationMonitor()
        monitor.update({1}, {1}, 0, True)

        self.assertEqual(
            monitor.update({1}, {1}, 4, False),
            {},
        )

        states = monitor.update({1}, {1}, 5, True)
        self.assertEqual(states[1].duration, 0)

    def test_visible_exit_clears_immediately(self):
        monitor = EscalationMonitor()
        monitor.update({1}, {1}, 0, True)

        self.assertEqual(
            monitor.update(set(), {1}, 0.1, True),
            {},
        )

    def test_missing_gap_freezes_duration_then_expires(self):
        monitor = EscalationMonitor()
        monitor.update({1}, {1}, 0, True)
        monitor.update({1}, {1}, 3, True)

        states = monitor.update(set(), set(), 3.5, True)

        self.assertEqual(states[1].duration, 3)
        self.assertTrue(states[1].missing)

        self.assertEqual(
            monitor.update(set(), set(), 4.1, True),
            {},
        )

    def test_short_missing_gap_retains_start(self):
        monitor = EscalationMonitor()
        monitor.update({1}, {1}, 0, True)
        monitor.update(set(), set(), 0.2, True)

        states = monitor.update({1}, {1}, 0.8, True)

        self.assertAlmostEqual(states[1].duration, 0.8)
        self.assertFalse(states[1].missing)

    def test_long_missing_gap_restarts_timer(self):
        monitor = EscalationMonitor()
        monitor.update({1}, {1}, 0, True)
        monitor.update(set(), set(), 0.5, True)

        states = monitor.update({1}, {1}, 2, True)

        self.assertEqual(states[1].duration, 0)

    def test_tracks_have_independent_timers(self):
        monitor = EscalationMonitor()
        monitor.update({1}, {1}, 0, True)

        states = monitor.update({1, 2}, {1, 2}, 4, True)

        self.assertEqual(states[1].level, "PERSISTENT")
        self.assertEqual(states[2].level, "ENTRY")

    def test_invalid_configuration_and_backwards_time(self):
        with self.assertRaises(ValueError):
            EscalationMonitor(6, 3)

        monitor = EscalationMonitor()
        monitor.update(set(), set(), 2, True)

        with self.assertRaises(ValueError):
            monitor.update(set(), set(), 1, True)


if __name__ == "__main__":
    unittest.main()