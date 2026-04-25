
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add src to path
# We assume the script is run from project root, so src is available
# But let's add it explicitly relative to script location or absolute path
project_root = r"d:\Trading\Trading_factors\DL_learning_project1\AlphaPROBE-master"
sys.path.append(os.path.join(project_root, "src"))

from alpha_knowledge.alpha_knowledge_trainer import AlphaKnowledgeLogger, AlphaKnowledgeTrainer

class TestLoggerFix(unittest.TestCase):
    def test_logger_attributes(self):
        # Mock dependencies
        test_data = MagicMock()
        target = MagicMock()
        args = MagicMock()
        log_dir = "test_log_dir"
        
        # Instantiate logger
        logger = AlphaKnowledgeLogger(test_data, target, log_dir, args)
        
        # Check if attributes are initialized
        self.assertTrue(hasattr(logger, 'llm_call_count'))
        self.assertTrue(hasattr(logger, 'llm_usage'))
        self.assertEqual(logger.llm_call_count, 0)
        self.assertEqual(logger.llm_usage, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        
        print("Logger attributes initialized correctly.")

if __name__ == '__main__':
    unittest.main()
