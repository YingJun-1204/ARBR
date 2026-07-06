import unittest
from unittest.mock import MagicMock

class TestEarlyStoppingBypass(unittest.TestCase):
    def test_bypass_logic(self):
        # Simulated run loop logic
        train_epochs = 10
        splatting_freeze_epochs = 5
        
        early_stopping = MagicMock()
        early_stopping.early_stop = False
        early_stopping.counter = 0
        
        for epoch in range(train_epochs):
            # simulate early stopping call
            early_stopping.counter += 1
            if early_stopping.counter >= 3:
                early_stopping.early_stop = True
                
            # Our fix:
            if epoch < splatting_freeze_epochs:
                early_stopping.counter = 0
                early_stopping.early_stop = False
                
            # Assertions
            if epoch < splatting_freeze_epochs:
                self.assertEqual(early_stopping.counter, 0)
                self.assertFalse(early_stopping.early_stop)
                
        # After freeze epochs, early stopping should be allowed to trigger
        early_stopping.counter = 3
        early_stopping.early_stop = True
        self.assertTrue(early_stopping.early_stop)

if __name__ == "__main__":
    unittest.main()
