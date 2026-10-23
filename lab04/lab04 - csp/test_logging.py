"""Logging regression checks; run with python -m unittest discover -s <this directory>."""
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from helper import save_log_result, summarize_experiment, load_run_history


class LoggingTests(unittest.TestCase):
    def test_log_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dict(board_size=4, best_board=np.array([1, 3, 0, 2]),
                          best_conflicts=0, iterations=2, history=[3, 1, 0],
                          runtime_ms=1.5, seed=1)
            options = dict(experiment_id='test', log_dir=tmp,
                           plot_run=False, verbose=False)
            first = save_log_result(result, **options)
            second = save_log_result(result, **options)
            self.assertEqual(second['run_number'], 2)
            history = load_run_history('test', first['run_id'], tmp)
            self.assertEqual(history.conflicts.tolist(), [3, 1, 0])
            self.assertNotIn('algorithm', history.columns)
            chunks = load_run_history('test', first['run_id'], tmp, chunksize=2)
            self.assertEqual(sum(len(chunk) for chunk in chunks), 3)
            runs = pd.read_csv(Path(tmp) / 'test/runs.csv')
            self.assertEqual(json.loads(runs.best_board.iloc[0]), [1, 3, 0, 2])
            with self.assertRaises(ValueError):
                save_log_result(result, run_number=1, **options)
            summary = summarize_experiment('test', log_dir=tmp, image_dir=tmp,
                                           show_plot=False, verbose=False)
            self.assertEqual(summary.runs.iloc[0], 2)
            metrics_only = save_log_result(result, save_history=False, **options)
            self.assertFalse((Path(tmp) / 'test/summary.csv').exists())
            with self.assertRaises(FileNotFoundError):
                load_run_history('test', metrics_only['run_id'], tmp)

    def test_legacy_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'old'
            directory.mkdir()
            path = directory / 'runs.csv'
            original = 'algorithm,board_size,run_number\nSA,4,1\n'
            path.write_text(original)
            with self.assertRaises(ValueError):
                save_log_result(dict(board_size=4, best_board=[1,3,0,2],
                                     best_conflicts=0, iterations=0, history=[0],
                                     runtime_ms=1, seed=1),
                                experiment_id='old', log_dir=tmp, plot_run=False)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(directory.rglob('*.gz')), [])


if __name__ == '__main__':
    unittest.main()
