# Useful for importing a few ch4 structures for testing:
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
min_ch4 = (torch.tensor([[6, 1, 1, 1, 1]]),
           torch.tensor([[[-3.2947064558, 0.6540086535, 0.4313881687],
                          [-2.2029514828, 0.6763552943, 0.4373590881],
                          [-3.6556797933, 0.7756014079, -0.5920264340],
                          [-3.6788104313, 1.4664338298, 1.0517860346],
                          [-3.6413841972, -0.3023558975, 0.8284340142]]]))
min_ch4 = [c.to(device) for c in min_ch4]
bad_ch4 = (torch.tensor([[6, 1, 1, 1, 1]]), 
           torch.tensor([[[-2.97117, -0.04326,  0.00000],
                          [-1.87917, -0.04326, -0.00000],
                          [-3.33517, -0.04326,  3.01880],
                          [-3.33517, -0.99976, -0.38088],
                          [-3.33517,  0.76485, -0.63791]]]))
bad_ch4 = [c.to(device) for c in bad_ch4]
really_bad_ch4 = (torch.tensor([[6, 1, 1, 1, 1]]),
                  torch.tensor([[[-3.03512,  1.26157,  0.26327],
                                 [-2.69779,  0.12977,  0.00000],
                                 [-4.12445, -0.59901,  0.69754],
                                 [-4.12445, -0.10992, -0.97991],
                                 [-4.12445,  1.09825,  0.28237]]]))
really_bad_ch4 = [c.to(device) for c in really_bad_ch4]
