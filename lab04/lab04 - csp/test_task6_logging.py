"""Regression checks for Task 6's shared compressed logging and resumable cache."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from helper import save_log_result
from task6_analysis import _cached_run, _read_source_history, _record


class Task6LoggingTests(unittest.TestCase):
    def test_resume_preserves_metrics_and_restart_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'task6_all_algorithms_v2'
            board = np.arange(8)
            algorithm = Mock(return_value={
                'board': board, 'conflicts': 1, 'time_ms': 2.5, 'iterations': 3,
                'evaluations': 20, 'history': [4, 1], 'full_history': [5, 4, 7, 1],
                'restart_boundaries': [0, 2], 'restart_history': [4, 1],
                'runs': 2, 'restarts': 1,
            })
            first = _cached_run(output, 'Restart HC', 8, 1, board, algorithm)
            resumed = _cached_run(output, 'Restart HC', 8, 1, board, algorithm)
            algorithm.assert_called_once()
            self.assertEqual(_record('Restart HC', 8, 1, board, first),
                             _record('Restart HC', 8, 1, board, resumed))
            self.assertEqual(first['full_history'], resumed['full_history'])
            self.assertEqual(first['restart_boundaries'], resumed['restart_boundaries'])
            self.assertEqual(resumed['evaluations'], 20)
            rows = pd.read_csv(output / 'runs.csv')
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows.history_csv.iloc[0].endswith('.csv.gz'))
            self.assertFalse((output / 'hc_runs').exists())
            with self.assertRaises(ValueError):
                _cached_run(output, 'Restart HC', 8, 1, board[::-1], algorithm)

    def test_source_history_new_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dict(board_size=4, best_board=[1, 3, 0, 2], best_conflicts=0,
                          history=[2, 0], iterations=1, runtime_ms=1, seed=1)
            save_log_result(result, experiment_id='sa', log_dir=tmp,
                            plot_run=False, verbose=False)
            source = Path(tmp) / 'sa'
            row = next(pd.read_csv(source / 'runs.csv').itertuples())
            self.assertEqual(_read_source_history(source, row).conflicts.tolist(), [2, 0])
            legacy = source / 'histories/legacy.csv'
            pd.DataFrame({'conflicts': [3, 0]}).to_csv(legacy, index=False)
            old = next(pd.DataFrame([{'run_id': 'legacy'}]).itertuples())
            self.assertEqual(_read_source_history(source, old).conflicts.tolist(), [3, 0])

    def test_proposal_history_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'cache'
            board = np.arange(8)
            algorithm = Mock(return_value={
                'board': board, 'conflicts': 1, 'time_ms': 1, 'iterations': 1,
                'history': [4, 1], 'proposal_history': [4, 4, 1],
            })
            _cached_run(output, 'Efficient HC', 8, 1, board, algorithm)
            cached = _cached_run(output, 'Efficient HC', 8, 1, board, algorithm)
            self.assertEqual(cached['proposal_history'], [4, 4, 1])
            algorithm.assert_called_once()


if __name__ == '__main__':
    unittest.main()
