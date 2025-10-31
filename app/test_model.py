#Ignore for now



# # unit tests for model

# import torch
# from .model import model_probability_opinion

# # =======================================================
# #                  Helper Function Tests
# # =======================================================

# def test_model_probability_opinion():
#     #Test the model_probability_opinion
#     test_list: list = [True, True, False, True, False]  # 2/5 False
#     assert model_probability_opinion(test_list) == 40.0
    
#     test_list = [True, True, True, True, True, True, True, True]
#     assert model_probability_opinion(test_list) == 0.0
    
#     test_list = [False, False]
#     assert model_probability_opinion(test_list) == 100.0
    
#     print("Test: 'test_model_probability_opinion' complete.")

# # =======================================================
# #                  Model Tests
# # =======================================================

# # Add additional model tests here as needed

# if __name__ == "__main__":
#     test_model_probability_opinion()
#     print("All tests passed!")