import os
import subprocess
import sys
import unittest

# /path/to/demos/antenna-selection/tests/test_antennas.py
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDemo(unittest.TestCase):
    def test_smoke(self):
        """run antennas.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'antennas.py')
        subprocess.check_output([sys.executable, demo_file])
