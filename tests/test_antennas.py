# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import unittest

# /path/to/demos/antenna-selection/tests/test_antennas.py
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDemo(unittest.TestCase):
    def test_antennas_smoke(self):
        """run antennas.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'antennas.py')
        subprocess.check_output([sys.executable, demo_file])

    def test_demo_smoke(self):
        """run demo.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'demo.py')
        subprocess.check_output([sys.executable, demo_file])

    def test_antenna_selection(self):
        """ Verify contents of output """
        
        demo_file = os.path.join(project_dir, 'antennas.py')
        output = subprocess.check_output([sys.executable, demo_file])
        output = str(output).upper()
        if os.getenv('DEBUG_OUTPUT'):
            print("Example output \n" + output)

        with self.subTest(msg="Verify if output contains 'Maximum independent set size found' \n"):
            self.assertIn("Maximum independent set size found".upper(), output)
        with self.subTest(msg="Verify if error string contains in output \n"):
            self.assertNotIn("ERROR", output)
        with self.subTest(msg="Verify if warning string contains in output \n"):
            self.assertNotIn("WARNING", output)