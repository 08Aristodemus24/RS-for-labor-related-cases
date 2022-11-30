import deepmatcher as dm
import torch

print("GPU:", torch.cuda.is_available())

train, validation, test = dm.data.process(path='sets', train='train.csv', validation='val.csv', test='test.csv',
                                          left_prefix='left_',
                                          right_prefix='right_',
                                          label_attr='label',
                                          id_attr='id')

model = dm.MatchingModel()

model.run_train(train, validation, best_save_path='./models/model_1.pth')

model.run_eval(test)
