"""Numbered Medallion directories must preserve existing public imports."""

import importlib
import unittest

from src.data import bronze, silver, gold
from src.data.bronze.models import BronzeRecord
from src.data.silver.models import SilverGame
from src.data.gold.models import PolicySample
from src.data.silver.data_pipeline import SilverPipeline


class NumberedDataPackageTests(unittest.TestCase):
    def test_aliases_point_to_numbered_packages(self):
        for alias, numbered, package in (
            ("bronze", "01_bronze", bronze),
            ("silver", "02_silver", silver),
            ("gold", "03_gold", gold),
        ):
            with self.subTest(alias=alias):
                self.assertIs(package, importlib.import_module(f"src.data.{numbered}"))

    def test_legacy_imports(self):
        self.assertIsNotNone(BronzeRecord)
        self.assertIsNotNone(SilverGame)
        self.assertIsNotNone(PolicySample)
        self.assertIsNotNone(SilverPipeline)


if __name__ == "__main__":
    unittest.main()
