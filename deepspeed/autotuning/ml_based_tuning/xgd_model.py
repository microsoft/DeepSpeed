import xgboost as xgb
import time
import numpy as np


class XGBoostCostModel():
    """XGBoost as cost model
    Parameters
    ----------
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
    log_interval: int, optional
        If is not none, the cost model will print training log every `log_interval` iterations.
    """
    def __init__(self, loss_type, log_interval=25):
        super(XGBoostCostModel, self).__init__()

        self.loss_type = loss_type
        self.log_interval = log_interval

        if loss_type == "reg":
            self.xgb_params = {
                "max_depth": 3, # TODO
                "gamma": 0.0001,
                "min_child_weight": 1, # TODO
                "subsample": 1.0, # TODO
                "eta": 0.3,
                "lambda": 1.00,
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
                "lambda": 1.00,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        self.bst = None
        self._sample_size = 0

    def fit(self, xs, ys):
        tic = time.time()

        x_train = np.array(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self._sample_size = len(x_train)

        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=10,
        )
        # print(f"Time: {time.time() - tic}")

    def predict(self, xs, output_margin=False):
        feas = np.array(xs)
        dtest = xgb.DMatrix(feas)
        return self.bst.predict(dtest, output_margin=output_margin)
