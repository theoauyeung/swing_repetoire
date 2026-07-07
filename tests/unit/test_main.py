import unittest

from src.main import testFunction

class TestSum(unittest.TestCase):
    def test_testFunction(self):
        """
        Test that it can sum a list of integers
        """
        self.assertTrue(testFunction())

if __name__ == '__main__':
    unittest.main()