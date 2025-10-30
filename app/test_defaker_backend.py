import app.defaker_backend as df
import torch
import torchvision
from unittest.mock import Mock, MagicMock


# =======================================================
#                  Helper Functions
# =======================================================    
def test_model_probability_opinion():
    test_list: list = [True, True, False, True, False] # 2/5 False
    assert df.model_probability_opinion(test_list) == 0.4
    test_list = [True, True, True, True, True, True, True, True]
    assert df.model_probability_opinion(test_list) == 0.0
    test_list = [False, False]
    assert df.model_probability_opinion(test_list) == 1.0
    print("Test: 'test_model_probability_opinion' complete.")


# =======================================================
#                      Models
# =======================================================