import numpy as np
import pandas as pd
import torch
from polara import get_movielens_data
from polara.preprocessing.dataframes import leave_one_out, reindex


def transform_indices(data, users, items):
    """
    Reindex columns that correspond to users and items.
    New index is contiguous starting from 0.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data to be reindexed.
    users : str
        The name of the column in `data` that contains user IDs.
    items : str
        The name of the column in `data` that contains item IDs.

    Returns
    -------
    pandas.DataFrame, dict
        The reindexed data and a dictionary with mapping between original IDs and the new numeric IDs.
        The keys of the dictionary are 'users' and 'items'.
        The values of the dictionary are pandas Index objects.

    Examples
    --------
    >>> data = pd.DataFrame({'customers': ['A', 'B', 'C'], 'products': ['X', 'Y', 'Z'], 'rating': [1, 2, 3]})
    >>> data_reindexed, data_index = transform_indices(data, 'customers', 'products')
    >>> data_reindexed
       users  items  rating
    0      0      0       1
    1      1      1       2
    2      2      2       3
    >>> data_index
    {
        'users': Index(['A', 'B', 'C'], dtype='object', name='customers'),
        'items': Index(['X', 'Y', 'Z'], dtype='object', name='products')
    }
    """
    data_index = {}
    for entity, field in zip(["users", "items"], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        data = data.assign(**{f"{field}": new_index})  # makes a copy of dataset!
    return data, data_index


def to_numeric_id(data, field):
    """
    This function takes in two arguments, data and field. It converts the data field
    into categorical values and creates a new contiguous index. It then creates an
    idx_map which is a renamed version of the field argument. Finally, it returns the
    idx and idx_map variables.
    """
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def custom_collate(data_list):
    first_state = []
    first_sampled_actions = []
    current_state = []
    current_action = []
    next_state = []
    next_sampled_actions = []
    rewards = []
    step_num = []
    has_next = []

    for fs, fsa, cs, ca, ns, nsa, rw, sn, hn in data_list:
        first_state.append(fs)
        first_sampled_actions.append(fsa)
        current_state.append(cs)
        current_action.append(ca)
        next_state.append(ns)
        next_sampled_actions.append(nsa)
        rewards.append(rw)
        step_num.append(sn)
        has_next.append(hn)

    return (
        torch.concat(first_state, dim=0),
        torch.concat(first_sampled_actions, dim=0),
        torch.concat(current_state, dim=0),
        torch.concat(current_action, dim=0),
        torch.concat(next_state, dim=0),
        torch.concat(next_sampled_actions, dim=0),
        torch.concat(rewards, dim=0),
        torch.concat(step_num, dim=0),
        torch.concat(has_next, dim=0),
    )


def get_dataset(
    validation_size=1024,
    test_size=5000,
    verbose=False,
    data_path=None,
    path=None,
    splitting="temporal_full",
    q=0.8,
):
    if type(path) == pd.core.frame.DataFrame:
        mldata = path
    elif isinstance(path, str) and path.endswith(".csv"):
        mldata = pd.read_csv(path)
    else:
        mldata = get_movielens_data(local_file=data_path, include_time=True).rename(
            columns={"movieid": "itemid"}
        )

    if splitting == "temporal_full":
        test_timepoint = mldata["timestamp"].quantile(q=q, interpolation="nearest")
        test_data_ = mldata.query("timestamp >= @test_timepoint")
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            "timestamp < @test_timepoint"
        )  # убрал userid not in @test_data_.userid.unique() and
        training_, data_index = transform_indices(
            train_data_.copy(), "userid", "itemid"
        )

        testset_ = reindex(test_data_, data_index["items"])

        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target="timestamp", sample_top=True, random_state=0
        )

        testset_valid, data_index = transform_indices(
            testset_valid_.copy(), "userid", "itemid"
        )

        testset_ = reindex(testset_, data_index["items"])
        holdout_valid = reindex(holdout_valid_, data_index["items"])
        holdout_valid = reindex(holdout_valid, data_index["users"]).sort_values(
            ["userid"]
        )
        if verbose:
            print(testset_.nunique())

        training = testset_valid.copy()

        validation_users = np.intersect1d(
            holdout_valid["userid"].unique(), testset_valid["userid"].unique()
        )
        if validation_size < len(validation_users):
            validation_users = np.random.choice(
                validation_users, size=validation_size, replace=False
            )
        testset_valid = testset_valid[
            testset_valid["userid"].isin(validation_users)
        ].sort_values(by=["userid", "timestamp"])
        holdout_valid = holdout_valid[holdout_valid["userid"].isin(validation_users)]

        testset_, holdout_ = leave_one_out(
            testset_, target="timestamp", sample_top=True, random_state=0
        )

        test_users = np.intersect1d(
            holdout_["userid"].unique(), testset_["userid"].unique()
        )
        if test_size < len(test_users):
            test_users = np.random.choice(test_users, size=test_size, replace=False)
        testset_ = testset_[testset_["userid"].isin(test_users)].sort_values(
            by=["userid", "timestamp"]
        )
        holdout_ = holdout_[holdout_["userid"].isin(test_users)].sort_values(["userid"])

    elif splitting == "temporal":
        test_timepoint = mldata["timestamp"].quantile(q=0.95, interpolation="nearest")
        test_data_ = mldata.query("timestamp >= @test_timepoint")
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            "userid not in @test_data_.userid.unique() and timestamp < @test_timepoint"
        )
        training, data_index = transform_indices(train_data_.copy(), "userid", "itemid")

        test_data = reindex(test_data_, data_index["items"])
        if verbose:
            print(test_data.nunique())
        testset_, holdout_ = leave_one_out(
            test_data, target="timestamp", sample_top=True, random_state=0
        )
        testset_valid_, holdout_valid_ = leave_one_out(
            testset_, target="timestamp", sample_top=True, random_state=0
        )

        userid = data_index["users"].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(), holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

    elif splitting == "leave-one-out":
        mldata, data_index = transform_indices(mldata.copy(), "userid", "itemid")
        training, holdout_ = leave_one_out(
            mldata, target="timestamp", sample_top=True, random_state=0
        )
        training_valid_, holdout_valid_ = leave_one_out(
            training, target="timestamp", sample_top=True, random_state=0
        )

        testset_valid_ = training_valid_.copy()
        testset_ = training.copy()
        training = training_valid_.copy()

        userid = data_index["users"].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(), holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

    else:
        raise ValueError

    if verbose:
        print(testset_valid.nunique())
        print(holdout_valid.shape)
    # assert holdout_valid.set_index('userid')['timestamp'].ge(
    #     testset_valid
    #     .groupby('userid')
    #     ['timestamp'].max()
    # ).all()

    data_description = dict(
        users=data_index["users"].name,
        items=data_index["items"].name,
        order="timestamp",
        n_users=len(data_index["users"]),
        n_items=len(data_index["items"]),
    )

    if verbose:
        print(data_description)

    return training, data_description, testset_valid, testset_, holdout_valid, holdout_
