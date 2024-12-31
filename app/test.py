import unittest
from datetime import datetime, timedelta

class TestAzureClients(unittest.TestCase):
    def test_timedelta(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        self.assertTrue(start_time < end_time)
