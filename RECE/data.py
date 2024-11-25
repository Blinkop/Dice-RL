from numba import njit
from numba.typed import List

from random import seed as set_seed
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex, leave_one_out

from RECE.sampler import prime_sampler_state, sample_unseen
from RECE.utils import transform_indices



def get_dataset(validation_size=1024, test_size=5000, verbose=False, data_path=None, path=None, splitting='temporal_full', q=0.8):
    if type(path) == pd.core.frame.DataFrame: 
        mldata = path
    else:
        mldata = get_movielens_data(local_file=data_path, include_time=True).rename(columns={'movieid': 'itemid'})
        

    if splitting == 'full':

        # айтемы, появившимися после q
        test_timepoint = mldata['timestamp'].quantile(
            q=q, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        
        print(test_data_.nunique())
        
        train_data_ = mldata.query(
            'timestamp < @test_timepoint') # убрал userid not in @test_data_.userid.unique() and 
        training_, data_index1 = transform_indices(train_data_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(test_data_, data_index1['items'])
        
        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )
        
        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(testset_, data_index['items'])
        holdout_valid = reindex(holdout_valid_, data_index['items'])
        
        print(testset_.nunique())
        
        # убираем из полного датасета интеракшены с айтемами, появившимися после q
        testset_valid, holdout_valid = leave_one_out(
        mldata, target='timestamp', sample_top=True, random_state=0
        )
        # training_valid_, holdout_valid_ = leave_one_out(
        #     training, target='timestamp', sample_top=True, random_state=0
        # )
        
        testset_valid = reindex(testset_valid, data_index1['items'])
        holdout_valid = reindex(holdout_valid, data_index1['items'])
        
        testset_valid = reindex(testset_valid, data_index['items'])
        holdout_valid = reindex(holdout_valid, data_index['items'])
        
        testset_ = testset_valid.copy()
        training = testset_valid.copy()

        userid = data_index['users'].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid[userid].unique(),
                holdout_valid[userid].unique()
            )
        )
        testset_valid = (
            testset_valid
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
                
        holdout_ = holdout_valid.copy() # просто что то поставил как холдаут
    
    elif splitting == 'temporal_full':

        test_timepoint = mldata['timestamp'].quantile(
            q=q, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            'timestamp < @test_timepoint') # убрал userid not in @test_data_.userid.unique() and 
        training_, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')

        testset_ = reindex(test_data_, data_index['items'])

        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )
        
        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(testset_, data_index['items'])
        holdout_valid = reindex(holdout_valid_, data_index['items'])
        holdout_valid = reindex(holdout_valid, data_index['users']).sort_values(['userid'])
        if verbose:
            print(testset_.nunique())

        training = testset_valid.copy()


        # validation_size = 1024
        validation_users = np.intersect1d(holdout_valid['userid'].unique(), testset_valid['userid'].unique())
        if validation_size < len(validation_users):
            validation_users  = np.random.choice(validation_users, size=validation_size, replace=False)
        testset_valid = testset_valid[testset_valid['userid'].isin(validation_users)].sort_values(by=['userid', 'timestamp'])
        holdout_valid = holdout_valid[holdout_valid['userid'].isin(validation_users)]

        testset_, holdout_ = leave_one_out(
            testset_, target='timestamp', sample_top=True, random_state=0
        )

        # test_size = 5000
        test_users = np.intersect1d(holdout_['userid'].unique(), testset_['userid'].unique())
        if test_size < len(test_users):
            test_users  = np.random.choice(test_users, size=test_size, replace=False)
        testset_ = testset_[testset_['userid'].isin(test_users)].sort_values(by=['userid', 'timestamp'])
        holdout_ = holdout_[holdout_['userid'].isin(test_users)].sort_values(['userid'])
        
        # holdout_ = testset_.copy() # просто что то поставил как хзолдаут, нам метрики не нужны для теста

    elif splitting == 'temporal_full_with_history':

        test_timepoint = mldata['timestamp'].quantile(
            q=q, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            'timestamp < @test_timepoint') # убрал userid not in @test_data_.userid.unique() and 
        training_, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')

        testset_ = reindex(test_data_, data_index['items'])
        
        ###
        test_user_idx = data_index['users'].get_indexer(testset_['userid'])
        is_new_user = test_user_idx == -1
        if is_new_user.any(): # track unseen users - to be used in warm-start regime
            new_user_idx, data_index['new_users'] = pd.factorize(testset_.loc[is_new_user, 'userid'])
            # ensure no intersection with train users index
            test_user_idx[is_new_user] = new_user_idx + len(data_index['users'])
        # assign new user index
        testset_.loc[:, 'userid'] = test_user_idx
        ###
        
        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )
        
        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(testset_, data_index['items'])
        ###
        test_user_idx = data_index['users'].get_indexer(testset_['userid'])
        is_new_user = test_user_idx == -1
        if is_new_user.any(): # track unseen users - to be used in warm-start regime
            new_user_idx, data_index['new_users'] = pd.factorize(testset_.loc[is_new_user, 'userid'])
            # ensure no intersection with train users index
            test_user_idx[is_new_user] = new_user_idx + len(data_index['users'])
        # assign new user index
        testset_.loc[:, 'userid'] = test_user_idx
        ###
        
        holdout_valid = reindex(holdout_valid_, data_index['items'])
        holdout_valid = reindex(holdout_valid, data_index['users'])
        
        if verbose:
            print(testset_.nunique())

        training = testset_valid.copy()

        # userid = data_index['users'].name
        # test_users = pd.Index(
        #     # ensure test users are the same across testing data
        #     np.intersect1d(
        #         testset_valid[userid].unique(),
        #         holdout_valid[userid].unique()
        #     )
        # )
        testset_valid = (
            testset_valid
            # reindex warm-start users for convenience
            # .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            # .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid
            # reindex warm-start users for convenience
            # .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            # .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        
        holdout_ = testset_.copy() # просто что то поставил как хзолдаут, нам метрики не нужны для теста
    
    elif splitting == 'temporal':
        test_timepoint = mldata['timestamp'].quantile(
            q=0.95, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            'userid not in @test_data_.userid.unique() and timestamp < @test_timepoint'
        )
        training, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')

        test_data = reindex(test_data_, data_index['items'])
        if verbose:
            print(test_data.nunique())
        testset_, holdout_ = leave_one_out(
            test_data, target='timestamp', sample_top=True, random_state=0
        )
        testset_valid_, holdout_valid_ = leave_one_out(
            testset_, target='timestamp', sample_top=True, random_state=0
        )
        
        userid = data_index['users'].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(),
                holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
    
        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )

    elif splitting == 'leave-one-out':
        mldata, data_index = transform_indices(mldata.copy(), 'userid', 'itemid')
        training, holdout_ = leave_one_out(
        mldata, target='timestamp', sample_top=True, random_state=0
        )
        training_valid_, holdout_valid_ = leave_one_out(
            training, target='timestamp', sample_top=True, random_state=0
        )

        testset_valid_ = training_valid_.copy()
        testset_ = training.copy()
        training = training_valid_.copy()

        userid = data_index['users'].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(),
                holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
    
        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
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
        users = data_index['users'].name,
        items = data_index['items'].name,
        order = 'timestamp',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
    )

    if verbose:
        print(data_description)

    return training, data_description, testset_valid, testset_, holdout_valid, holdout_

def no_sample(user_items, maxlen, pad_token):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.empty((maxlen, 1))

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]

    return seq, pos, neg

def sample_dross(all_items, user_items, maxlen, pad_token, n_neg_samples, random_state):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]

    neg = random_state.choice(all_items[~np.isin(all_items, user_items)], n_neg_samples, replace=False)

    return seq, pos, neg

def sample_with_rep(user_items, maxlen, pad_token, n_neg_samples, itemnum, random_state):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.full((n_neg_samples, maxlen), pad_token, dtype=np.int32)

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]
    neg[:, -n_user_items:] = random_state.randint(0, itemnum, (n_neg_samples, n_user_items))

    return seq, pos, neg

@njit
def sample_without_rep(user_items, maxlen, pad_token, n_neg_samples, itemnum, seed):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.full((maxlen, n_neg_samples), pad_token, dtype=np.int32)

    hist_items_counter = 1
    nxt = user_items[-1]
    idx = maxlen - 1

    set_seed(seed)

    ts_ = list(set(user_items))

    for i in user_items[-2::-1]:
        seq[idx] = i
        pos[idx] = nxt

        state = prime_sampler_state(itemnum, ts_)
        remaining = itemnum - len(ts_)
        
        sample_unseen(n_neg_samples, state, remaining, neg[idx])

        nxt = i
        idx -= 1
        hist_items_counter += 1
        if idx == -1:
            break
        
    neg = np.swapaxes(neg, 0, 1)
    return seq, pos, neg

class SequentialDataset(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen, seed, n_neg_samples=1, sampling='without_rep', pad_token=None):
        super().__init__()
        self.user_train = user_train

        self.valid_users = [user for user in range(usernum) if len(user_train.get(user, [])) > 1]

        self.usernum = len(self.valid_users)

        self.itemnum = itemnum
        self.maxlen = maxlen
        self.seed = seed
        self.n_neg_samples = n_neg_samples
        self.sampling = sampling
        
        if self.sampling == 'dross':
            self.all_items = np.arange(self.itemnum, dtype=np.int32)

        self.pad_token = pad_token

        self.random_state = np.random.RandomState(self.seed)

    def __len__(self):
        return self.usernum
    
    def __getitem__(self, idx):
        user = self.valid_users[idx]
        user_items = List()
        [user_items.append(x) for x in self.user_train[user]]

        if self.sampling == 'with_rep':
            seq, pos, neg = sample_with_rep(user_items, self.maxlen, self.pad_token,
                                            self.n_neg_samples, self.itemnum,
                                            self.random_state)
        elif self.sampling == 'without_rep':
            seq, pos, neg = sample_without_rep(user_items, self.maxlen, self.pad_token,
                                               self.n_neg_samples, self.itemnum,
                                               self.random_state.randint(np.iinfo(int).min, np.iinfo(int).max))
        elif self.sampling == 'dross':
            seq, pos, neg = sample_dross(self.all_items, user_items, self.maxlen, self.pad_token,
                                         self.n_neg_samples,
                                         self.random_state)
        elif self.sampling == 'no_sampling':
            seq, pos, neg = no_sample(user_items, self.maxlen, self.pad_token)
        else:
            raise NotImplementedError()

        return user, seq, pos, neg

def sample_f(x, all_items, n_samples):
    seq = list(zip(*x))[0]
    seq_compl = all_items - set(seq)
    sampled_items = np.random.choice(list(seq_compl), n_samples, replace=False).tolist()

    return x + list(zip(sampled_items, [0] * n_samples))

def data_to_sequences_rating(data, data_description, n_neg_samples=-1):
    userid = data_description['users']
    itemid = data_description['items']
    data['itemid_rating'] = data[[itemid, 'rating']].apply(tuple, axis=1)
    sequences = (
        data.sort_values([userid, data_description['order']])
        .groupby(userid, sort=False)['itemid_rating'].apply(list)
    )
    if n_neg_samples > 0:
        n_items = data_description['n_items']
        all_items = set(range(n_items))
        sequences = sequences.apply(lambda x: sample_f(x, all_items, n_neg_samples))

    return sequences

def data_to_sequences(data, data_description):
    userid = data_description['users']
    itemid = data_description['items']

    sequences = (
        data.sort_values([userid, data_description['order']])
        .groupby(userid, sort=False)[itemid].apply(list)
    )
    return sequences

if __name__ == '__main__':
    get_dataset()
