from ml_collections import config_dict

def get_config(alg_type):
    config_e = {
        "SASRec0": config_dict.ConfigDict(
            {
                "gen_model": "prepare_sasrec_model",
                "params": config_dict.ConfigDict(
                    {
                        "manual_seed": 123,
                        "sampler_seed": 123,
                        "item_emb_svd": None,
                        "init_emb_svd": None,
                        "use_lin_layer": False,
                        "num_epochs": 100,
                        "maxlen": 100,
                        "hidden_units": 64,
                        "dropout_rate": 0.3,
                        "num_blocks": 2,
                        "num_heads": 1,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "fwd_type": 'ce',
                        "l2_emb": 0,
                        "patience": 10,
                        "skip_epochs": 1,
                        "n_neg_samples": 0,
                        "sampling": "no_sampling"
                    }
                ),
                "chkpt_path": "./saved_models/model_e0.pt"
            }
        ),

        "SASRec1": config_dict.ConfigDict(
            {
                "gen_model": "prepare_sasrec_model",
                "params": config_dict.ConfigDict(
                    {
                        "manual_seed": 123,
                        "sampler_seed": 123,
                        "item_emb_svd": None,
                        "init_emb_svd": None,
                        "use_lin_layer": False,
                        "num_epochs": 10,
                        "maxlen": 100,
                        "hidden_units": 64,
                        "dropout_rate": 0.3,
                        "num_blocks": 2,
                        "num_heads": 1,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "fwd_type": 'ce',
                        "l2_emb": 0,
                        "patience": 10,
                        "skip_epochs": 1,
                        "n_neg_samples": 0,
                        "sampling": 'no_sampling'
                    }
                ),
                "chkpt_path": "./saved_models/model_e1.pt"
            }
        ),

        "SASRec2": config_dict.ConfigDict(
            {
                "gen_model": "prepare_sasrec_model",
                "params": config_dict.ConfigDict(
                    {
                        "manual_seed": 123,
                        "sampler_seed": 123,
                        "item_emb_svd": None,
                        "init_emb_svd": None,
                        "use_lin_layer": False,
                        "num_epochs": 20,
                        "maxlen": 100,
                        "hidden_units": 32,
                        "dropout_rate": 0.9,
                        "num_blocks": 2,
                        "num_heads": 1,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "fwd_type": 'ce',
                        "l2_emb": 0,
                        "patience": 10,
                        "skip_epochs": 1,
                        "n_neg_samples": 0,
                        "sampling": 'no_sampling'
                    }
                ),
                "chkpt_path": "./saved_models/model_e2.pt"
            }
        ),

        "CQL": config_dict.ConfigDict(
            {
                "gen_model": "prepare_cql_model",
                "params": config_dict.ConfigDict(
                    {
                        "orthogonal_init": True,
                        "q_n_hidden_layers": 1,
                        "qf_lr": 3e-4,
                        "batch_size": sasrec_config['batch_size'],
                        "device": "cuda",
                        "bc_steps": 100000,
                        "cql_alpha": 100.0,
                        "env": "MovieLens",
                        "project": "CQL-SASREC",
                        "group": "CQL-SASREC",
                        "name": "CQL"
                        #cql_negative_samples = 10
                    }
                ),
                "chkpt_path": "./saved_models/model_e2.pt"
            }
        )
    }[alg_type]

    config_D = config_dict.ConfigDict(
        {
            "gen_model": "prepare_sasrec_model",
            "params": config_dict.ConfigDict(
                {
                    "manual_seed": 123,
                    "sampler_seed": 123,
                    "item_emb_svd": None,
                    "init_emb_svd": None,
                    "use_lin_layer": False,
                    "num_epochs": 100,
                    "maxlen": 100,
                    "hidden_units": 64,
                    "dropout_rate": 0.3,
                    "num_blocks": 2,
                    "num_heads": 1,
                    "batch_size": 128,
                    "learning_rate": 1e-3,
                    "fwd_type": 'ce',
                    "l2_emb": 0,
                    "patience": 10,
                    "skip_epochs": 1,
                    "n_neg_samples": 0,
                    "sampling": "no_sampling"
                }
            ),
            "chkpt_path": "./saved_models/model_D_sasrec.pt"
        }
    )

    config = config_dict.ConfigDict(
        {
            "config_e": config_e,
            "config_D": config_D
        }
    )

    config.optim_conf = config_dict.ConfigDict(
        {
            "lr": 3e-4,
            "weight_decay": 1e-4
        }
    )

    config.fqe_params = config_dict.ConfigDict(
        {
            "n_epochs": 200,
            "hidden_size": 512,
            "bs": 512
        }
    )

    config.subseq_len = 3
    config.subseq_len_val = 3
    config.n_neg_samples = 1
    config.gamma = 0.98
    config.device = "cuda"
    config.rank = -1
    config.seed = 42
    config.binary_rew = False

    config.alg_type = alg_type
    # config.values_path = f"./saved_values/values0_pen={config.optim_conf.weight_decay},"\
    #                      f"bs={config.fqe_params.bs},"\
    #                      f"nneg={config.n_neg_samples},"\
    #                      f"subseq_len={config.subseq_len}.npy" #in cmd

    # config.fqe_w_path = f"saved_fqes/fqe0_pen={config.optim_conf.weight_decay},"\
    #                     f"bs={config.fqe_params.bs},"\
    #                     f"nneg={config.n_neg_samples},"\
    #                     f"subseq_len={config.subseq_len}.pt"

    return config






