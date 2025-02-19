# -*- coding: utf-8 -*-

import argparse
import functools
import datetime

import json
import os
import pickle
import time
import traceback

from gym import register
import gym
import torch

from torch import nn
from core.inputs import SparseFeatP
from core.RZ_model_pairwise import UserModel_Pairwise
from core.util import compute_exposure_each_user, get_similarity_mat, \
    compute_exposure_effect_kuaishouRec, load_static_validate_data_kuaishou
from core.kuairand_utils import negative_sampling
from deepctr_torch.inputs import DenseFeat
import pandas as pd
import numpy as np

from core.static_dataset import StaticDataset
from torch.utils.data import DataLoader

import logzero
from logzero import logger

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

# from environments.KuaishouRec.env.data_handler import get_training_item_domination
# from environments.KuaishouRec.env.kuaishouEnv import KuaishouEnv

from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv

from evaluation import test_kuaishou, test_static_model_in_RL_env
# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_Update

DATAPATH = "environments/KuaiRand_Pure/data"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--env", type=str, default='KuaiRand-v0')

    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--not_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=False)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)
    parser.add_argument('--lambda_ab', default=10, type=float)

    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--is_ucb', dest='is_ucb', action='store_true')
    parser.add_argument('--no_ucb', dest='is_ucb', action='store_false')
    parser.set_defaults(is_ucb=False)


    parser.add_argument("--num_trajectory", type=int, default=200)
    parser.add_argument("--force_length", type=int, default=10)
    parser.add_argument("--top_rate", type=float, default=0.8)

    parser.add_argument("--feature_dim", type=int, default=4)  #16
    parser.add_argument("--entity_dim", type=int, default=4)   #16
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--dnn', default=(128, 128), type=int, nargs="+")
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    # # env:
    parser.add_argument('--leave_threshold', default=0, type=float)
    parser.add_argument('--num_leave_compute', default=10, type=int)
    # exposure parameters:
    parser.add_argument('--tau', default=1000, type=float)

    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=True)
    parser.add_argument("--message", type=str, default="UserModel1")


    #add infos
    parser.add_argument("--yfeat", type=str, default='is_click')
    parser.add_argument('--max_turn', default=30, type=int)
    parser.add_argument('--neg_K', default=1, type=int)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    args = parser.parse_known_args()[0]
    return args

########## add function########

def get_features(is_userinfo=None):
    user_features = ["user_id", 'user_active_degree', 'is_live_streamer', 'is_video_author',
                        'follow_user_num_range',
                        'fans_user_num_range', 'friend_user_num_range', 'register_days_range'] \
                    + [f'onehot_feat{x}' for x in range(18)]
    if not is_userinfo:
        user_features = ["user_id"]
    item_features = ["item_id"] + ["feat" + str(i) for i in range(3)] + ["duration_normed"]
    reward_features = ["is_click"]

    return user_features, item_features, reward_features

def get_training_data(env):
    from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
    df_train, df_user, df_item, list_feat = KuaiRandEnv.get_df_kuairand("train_processed.csv")
    return df_train, df_user, df_item, list_feat

def get_val_data(env):
    from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
    df_val, df_user_val, df_item_val, list_feat = KuaiRandEnv.get_df_kuairand("test_processed.csv")
    return df_val, df_user_val, df_item_val, list_feat

def get_training_item_domination():
    item_feat_domination = None
    from environments.KuaiRand_Pure.env.KuaiRand import KuaiRandEnv
    item_feat_domination = KuaiRandEnv.get_domination()
    return item_feat_domination

def get_xy_columns(args, df_data, df_user, df_item, user_features, item_features, entity_dim, feature_dim):
    if args.env == "KuaiRand-v0" or args.env == "KuaiEnv-v0":
        feat = [x for x in df_item.columns if x[:4] == "feat"]
        x_columns = [SparseFeatP("user_id", df_data['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim, padding_idx=0) for col in
                     user_features[1:]] + \
                    [SparseFeatP("item_id", df_data['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(x,
                                 df_item[feat].max().max() + 1,
                                 embedding_dim=feature_dim,
                                 embedding_name="feat",  # Share the same feature!
                                 padding_idx=0  # using padding_idx in embedding!
                                 ) for x in feat] + \
                    [DenseFeat("duration_normed", 1)]

    else:
        x_columns = [SparseFeatP("user_id", df_data['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_user[col].max() + 1, embedding_dim=feature_dim) for col in user_features[1:]] + \
                    [SparseFeatP("item_id", df_data['item_id'].max() + 1, embedding_dim=entity_dim)] + \
                    [SparseFeatP(col, df_item[col].max() + 1, embedding_dim=feature_dim) for col in item_features[1:]]

    ab_columns = [SparseFeatP("alpha_u", df_data['user_id'].max() + 1, embedding_dim=1)] + \
                 [SparseFeatP("beta_i", df_data['item_id'].max() + 1, embedding_dim=1)]

    y_columns = [DenseFeat("y", 1)]
    return x_columns, y_columns, ab_columns


def construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features):
    user_ids = np.unique(dataset_val.x_numpy[:, dataset_val.user_col].astype(int))

    # # user_ids = random.sample(user_ids.tolist(),100)
    # user_ids = user_ids[:10000] # todo: for speeding up, we only use 10000 users for visual the bars.
    # logzero.logger.info("#####################\nNote that we use only 10000 users for static evaluation!!\n#####################")
    item_ids = np.unique(dataset_val.x_numpy[:, dataset_val.item_col].astype(int))

    df_user_complete = pd.DataFrame(
        df_user.loc[user_ids].reset_index()[user_features].to_numpy().repeat(len(item_ids), axis=0),
        columns=df_user.reset_index()[user_features].columns)
    df_item_complete = pd.DataFrame(np.tile(df_item.loc[item_ids].reset_index()[item_features], (len(user_ids), 1)),
                                    columns=df_item.loc[item_ids].reset_index()[item_features].columns)

    df_x_complete = pd.concat([df_user_complete, df_item_complete], axis=1)
    return df_x_complete


def compute_normed_reward(model, dataset_val, df_user, df_item, user_features, item_features, x_columns, y_columns):
    #df_x_complete得到所有用户-项目组合的特征，即使这些组合在原始验证集中并不存在
    df_x_complete = construct_complete_val_x(dataset_val, df_user, df_item, user_features, item_features)
    #n_user, n_item = df_x_complete[["user_id", "item_id"]].nunique()

    print("predict all users' rewards on all items")

    dataset_um = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_um.compile_dataset(df_x_complete, pd.DataFrame(np.zeros([len(df_x_complete), 1]), columns=["y"]))

    num_user = len(df_x_complete["user_id"].unique())
    num_item = len(df_x_complete["item_id"].unique())

    predict_all_cat = model.predict_data(dataset_um, batch_size=2048).reshape([-1])

    if num_user != df_x_complete["user_id"].max() + 1:
        assert num_item != df_x_complete["item_id"].max() + 1
        lbe_user = LabelEncoder()
        lbe_item = LabelEncoder()

        lbe_user.fit(df_x_complete["user_id"])
        lbe_item.fit(df_x_complete["item_id"])

        predict_mat = csr_matrix(
            (
            predict_all_cat, (lbe_user.transform(df_x_complete["user_id"]), lbe_item.transform(df_x_complete["item_id"]))),
            shape=(num_user, num_item)).toarray()

    else:
        assert num_item == df_x_complete["item_id"].max() + 1
        predict_mat = csr_matrix(
            (predict_all_cat, (df_x_complete["user_id"], df_x_complete["item_id"])),
            shape=(num_user, num_item)).toarray()
    
    return predict_mat


def process_logit(y_deepfm_pos, score, alpha_u=None, beta_i=None, args=None):
    if alpha_u is not None:
        score_new = score * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        score_new = score
        loss_ab = 0
    loss_ab = args.lambda_ab * loss_ab
    y_weighted = 1 / (1 + score_new) * y_deepfm_pos
    return y_weighted, loss_ab

def loss_pointwise_negative(y, y_deepfm_pos, y_deepfm_neg, score, alpha_u=None, beta_i=None, args=None, log_var=None,
                            log_var_neg=None):
    y_weighted, loss_ab = process_logit(y_deepfm_pos, score, alpha_u=alpha_u, beta_i=beta_i, args=args)

    if log_var is not None:
        inv_var = torch.exp(-log_var)
        inv_var_neg = torch.exp(-log_var_neg)
        loss_var_pos = log_var.sum()
        loss_var_neg = log_var_neg.sum()
    else:
        inv_var = 1
        inv_var_neg = 1
        loss_var_pos = 0
        loss_var_neg = 0

    loss_y = (((y_weighted - y) ** 2) * inv_var).sum()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2) * inv_var_neg).sum()

    loss = loss_y + loss_y_neg + loss_ab + loss_var_pos + loss_var_neg
    return loss

################## add end ###################

def load_dataset_train(args, user_features, item_features, reward_features, tau, entity_dim, feature_dim,
                       MODEL_SAVE_PATH, DATAPATH):
    df_train, df_user, df_item, list_feat = get_training_data(args.env)

    assert user_features[0] == "user_id"
    assert item_features[0] == "item_id"
    df_user = df_user[user_features[1:]]
    df_item = df_item[item_features[1:]]

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_train, df_user, df_item, user_features, item_features,
                                                      entity_dim, feature_dim)

    neg_in_train = True if args.env == "KuaiRand-v0" and reward_features[0] != "watch_ratio_normed" else False
    neg_in_train = False  # todo: test for kuairand

    df_pos, df_neg = negative_sampling(df_train, df_item, df_user, reward_features[0],
                                       is_rand=True, neg_in_train=neg_in_train, neg_K=args.neg_K)

    df_x = df_pos[user_features + item_features]
    if reward_features[0] == "hybrid":  # for kuairand
        a = df_pos["long_view"] + df_pos["is_like"] + df_pos["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
        df_pos["hybrid"] = df_y["hybrid"]
    else:
        df_y = df_pos[reward_features]

    df_x_neg = df_neg[user_features + item_features]
    df_x_neg = df_x_neg.rename(columns={k: k + "_neg" for k in df_x_neg.columns.to_numpy()})

    df_x_all = pd.concat([df_x, df_x_neg], axis=1)

    if tau == 0:
        exposure_pos = np.zeros([len(df_x_all), 1])
    else:
        timestamp = df_pos['time_ms']  #timestamp
        exposure_pos = compute_exposure_effect_kuaishouRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH, is_kuairand=True)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, exposure_pos)

    return dataset, df_user, df_item, x_columns, y_columns, ab_columns

def load_dataset_val(args, user_features, item_features, reward_features, entity_dim, feature_dim):
    df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)

    assert user_features[0] == "user_id"
    assert item_features[0] == "item_id"
    df_user_val = df_user_val[user_features[1:]]
    df_item_val = df_item_val[item_features[1:]]

    df_x = df_val[user_features + item_features]
    if reward_features[0] == "hybrid":  # for kuairand
        a = df_val["long_view"] + df_val["is_like"] + df_val["is_click"]
        df_y = a > 0
        df_y = pd.DataFrame(df_y, dtype=int, columns=["hybrid"])
    else:
        df_y = df_val[reward_features]

    x_columns, y_columns, ab_columns = get_xy_columns(args, df_val, df_user_val, df_item_val, user_features,
                                                      item_features,
                                                      entity_dim, feature_dim)

    dataset_val = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_val.compile_dataset(df_x, df_y)

    dataset_val.set_df_item_val(df_item_val)
    dataset_val.set_df_user_val(df_user_val)

    assert dataset_val.x_columns[0].name == "user_id"
    dataset_val.set_user_col(0)
    assert dataset_val.x_columns[len(user_features)].name == "item_id"
    dataset_val.set_item_col(len(user_features))

    if not any(df_y.to_numpy() % 1):  # 整数
        # make sure the label is binary

        df_binary = pd.concat([df_val[["user_id", "item_id"]], df_y], axis=1)
        df_ones = df_binary.loc[df_binary[reward_features[0]] > 0]
        ground_truth = df_ones[["user_id", "item_id"] + reward_features].groupby("user_id").agg(list)
        ground_truth.rename(columns={"item_id": "item_id", reward_features[0]: "y"}, inplace=True)

        # for ranking purpose.
        threshold = 1
        index = ground_truth["y"].map(lambda x: [True if i >= threshold else False for i in x])
        df_temp = pd.DataFrame(index)
        df_temp.rename(columns={"y": "ind"}, inplace=True)
        df_temp["y"] = ground_truth["y"]
        df_temp["true_id"] = ground_truth["item_id"]
        df_true_id = df_temp.apply(lambda x: np.array(x["true_id"])[x["ind"]].tolist(), axis=1)
        df_true_y = df_temp.apply(lambda x: np.array(x["y"])[x["ind"]].tolist(), axis=1)

        #if args.is_binarize:
        df_true_y = df_true_y.map(lambda x: [1] * len(x))

        ground_truth_revise = pd.concat([df_true_id, df_true_y], axis=1)
        ground_truth_revise.rename(columns={0: "item_id", 1: "y"}, inplace=True)
        dataset_val.set_ground_truth(ground_truth_revise)

        if args.all_item_ranking:
            dataset_val.set_all_item_ranking_in_evaluation(args.all_item_ranking)

            df_x_complete = construct_complete_val_x(dataset_val, df_user_val, df_item_val, user_features,
                                                     item_features)
            df_y_complete = pd.DataFrame(np.zeros(len(df_x_complete)), columns=df_y.columns)

            dataset_complete = StaticDataset(x_columns, y_columns, num_workers=4)
            dataset_complete.compile_dataset(df_x_complete, df_y_complete)
            dataset_val.set_dataset_complete(dataset_complete)

    return dataset_val, df_user_val, df_item_val

def main(args):
    args.entity_dim = args.feature_dim
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)

    create_dirs = [os.path.join(".", "saved_models"),
                   os.path.join(".", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    logger.info(json.dumps(vars(args), indent=2))

    # %% 2. Prepare Envs

    # mat, lbe_user, lbe_photo, list_feat, df_photo_env, df_dist_small = KuaiRandEnv.load_mat()
    # register(
    #     id=args.env,  # 'KuaishouEnv-v0',
    #     entry_point='environments.KuaishouRec.env.kuaishouEnv:KuaishouEnv',
    #     kwargs={"mat": mat,
    #             "lbe_user": lbe_user,
    #             "lbe_photo": lbe_photo,
    #             "num_leave_compute": args.num_leave_compute,
    #             "leave_threshold": args.leave_threshold,
    #             "list_feat": list_feat,
    #             "df_photo_env": df_photo_env,
    #             "df_dist_small": df_dist_small}
    # )
    # env = gym.make(args.env)

    mat, list_feat, mat_distance = KuaiRandEnv.load_mat(args.yfeat, read_user_num=None)
    kwargs_um = {"yname": args.yfeat,
                     "mat": mat,
                     "mat_distance": mat_distance,
                     "list_feat": list_feat,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn}
    env = KuaiRandEnv(**kwargs_um)


    # %% 3. Prepare dataset
    user_features, item_features, reward_features = get_features(False)
    # 训练集
    # static_dataset, x_columns, y_columns, ab_columns = load_dataset_kuaishou(args.tau, args.entity_dim,
    #                                                                          args.feature_dim,
    #                                                                          MODEL_SAVE_PATH)
    dataset_train, df_user, df_item, x_columns, y_columns, ab_columns = \
        load_dataset_train(args, user_features, item_features, reward_features,
                           args.tau, args.entity_dim, args.feature_dim, MODEL_SAVE_PATH, DATAPATH)
    
    if not args.is_ab:
        ab_columns = None

    # 验证集
    # dataset_val = load_static_validate_data_kuaishou(args.entity_dim, args.feature_dim, DATAPATH)
    dataset_val, df_user_val, df_item_val = load_dataset_val(args, user_features, item_features, reward_features,
                                                             args.entity_dim, args.feature_dim)

    # %% 4. Setup model
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    SEED = 2022
    task = "binary"  #"regression"
    task_logit_dim = 1
    model = UserModel_Pairwise(x_columns, y_columns, task, task_logit_dim,
                           dnn_hidden_units=args.dnn, seed=SEED, l2_reg_dnn=args.l2_reg_dnn,
                           device=device, ab_columns=ab_columns)

    model.compile(optimizer="adam",
                  # loss_dict=task_loss_dict,
                  loss_func = loss_kuaiRand_pairwise,  # loss_kuai_pairwise,
                  metric_fun={"mae": lambda y, y_predict: nn.functional.l1_loss(torch.from_numpy(y),
                                                                                torch.from_numpy(y_predict)).numpy(),
                              "mse": lambda y, y_predict: nn.functional.mse_loss(torch.from_numpy(y),
                                                                                 torch.from_numpy(y_predict)).numpy()},
                  metrics=None)  # No evaluation step at offline stage

    # model.compile_RL_test(
    #     functools.partial(test_kuaishou, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax, epsilon=args.epsilon, is_ucb=args.is_ucb))
    item_feat_domination = get_training_item_domination()
    model.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=args.is_ucb, need_transform=False,
                          num_trajectory=args.num_trajectory, item_feat_domination=item_feat_domination,
                          force_length=args.force_length, top_rate=args.top_rate))

    # %% 5. Learn model
    history = model.fit_data(dataset_train, dataset_val,
                             batch_size=args.batch_size, epochs=args.epoch,
                             callbacks=[LoggerCallback_Update(logger_path)])
    logger.info(history.history)

    model_parameters = {"feature_columns": x_columns, "y_columns": y_columns, "task": task,
                        "task_logit_dim": task_logit_dim, "dnn_hidden_units": args.dnn, "seed": SEED, "device": device,
                        "ab_columns": ab_columns}

    model_parameter_path = os.path.join(MODEL_SAVE_PATH,
                                        "{}_params_{}.pickle".format(args.user_model_name, args.message))
    with open(model_parameter_path, "wb") as output_file:
        pickle.dump(model_parameters, output_file)

    normed_mat = compute_normed_reward(model, dataset_val, df_user, df_item, user_features, item_features, x_columns, y_columns)
    mat_save_path = os.path.join(MODEL_SAVE_PATH, "normed_mat-{}.pickle".format(args.message))
    with open(mat_save_path, "wb") as f:
        pickle.dump(normed_mat, f)

    #  To cpu
    model = model.cpu()
    model.linear_model.device = "cpu"
    model.linear.device = "cpu"
    # for linear_model in user_model.linear_model_task:
    #     linear_model.device = "cpu"

    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.user_model_name, args.message))
    torch.save(model.state_dict(), model_save_path)

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)


sigmoid = nn.Sigmoid()
def loss_kuaishou_pairwise(y, y_deepfm_pos, y_deepfm_neg, exposure,  alpha_u=None, beta_i=None):

    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos

    loss_y = ((y_exposure - y) ** 2).mean()
    bpr_click = - sigmoid(y_deepfm_pos - y_deepfm_neg).log().mean()

    loss = loss_y + bpr_click + args.lambda_ab * loss_ab

    return loss


def loss_kuai_pairwise(y, y_deepfm_pos, y_deepfm_neg, exposure,  alpha_u=None, beta_i=None):

    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos

    loss_y = ((y_exposure - y) ** 2).mean()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2)).mean()

    loss = loss_y + loss_y_neg + args.lambda_ab * loss_ab

    return loss

def loss_kuaiRand_pairwise(y, y_deepfm_pos, y_deepfm_neg, exposure,  alpha_u=None, beta_i=None):

    if alpha_u is not None:
        exposure_new = exposure * alpha_u * beta_i
        loss_ab = ((alpha_u - 1) ** 2).mean() + ((beta_i - 1) ** 2).mean()
    else:
        exposure_new = exposure
        loss_ab = 0

    y_exposure = 1 / (1 + exposure_new) * y_deepfm_pos

    loss_y = ((y_exposure - y) ** 2).sum()   # mean()
    loss_y_neg = (((y_deepfm_neg - 0) ** 2)).sum()

    loss = loss_y + loss_y_neg + loss_ab

    return loss


if __name__ == '__main__':
    args = get_args()
    try:
        main(args)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
