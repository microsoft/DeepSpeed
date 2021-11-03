import numpy as np
import xgboost as xgb

from .utils import *


class XGBoostCostModel():
    def __init__(self,
                 loss_type,
                 keys=None,
                 num_threads=None,
                 log_interval=25,
                 upper_model=None):

        self.loss_type = loss_type
        self.keys = keys
        self.bst = None

        if loss_type == "reg":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.0,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif loss_type == "rank":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.0,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        if num_threads:
            self.xgb_params["nthread"] = num_threads

    def get_feature(self, xs):
        configs = []
        for x in xs:
            knob = dict_to_feature(x, self.keys)
            configs.append(knob)

        x_train = np.array(configs, dtype=np.float32)
        return x_train

    def fit(self, xs, ys):
        x_train = self.get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-9)

        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])

        self.bst = xgb.train(self.xgb_params, dtrain)

    def predict(self, xs):

        features = self.get_feature(xs)
        dtest = xgb.DMatrix(features)
        if self.bst:
            return self.bst.predict(dtest)
