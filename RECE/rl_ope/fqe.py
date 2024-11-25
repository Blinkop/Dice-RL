from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_size=128):
        super().__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        logits = self.model(x)

        return logits


class RLDatasetOnline(Dataset):
    def __init__(self, config):
        self.states = config["states"]

        if "next_states" in config:
            self.next_states = config["next_states"]

        self.actions = config["actions"]
        self.rewards = config["rewards"]
        self.full_sequences = list(config["full_sequences"].items())
        self.n = config["n"]
        self.pad_token = config["pad_token"]

        if "n_neg_samples" in config:
            self.n_neg_samples = config["n_neg_samples"]

        if "all_actions" in config:
            self.all_actions = config["all_actions"]

        self.n_subseqs = []
        for _, h in self.full_sequences:
            self.n_subseqs.append(len(h) - 1)

        self.pref_sum_n_subseqs = np.cumsum(self.n_subseqs)

    def __len__(self):
        return np.sum(self.n_subseqs)

    def get_state_action(self, idx):
        seq_n = np.searchsorted(self.pref_sum_n_subseqs, idx, side='right')
        seq = self.full_sequences[seq_n][1]

        subseq_n = idx - self.pref_sum_n_subseqs[seq_n - 1] if seq_n else idx
        subseq_n = self.n_subseqs[seq_n] - subseq_n
        action = seq[subseq_n][0]
        assert action == self.actions[idx]
        state_seq = torch.tensor(seq[max(subseq_n - self.n, 0):subseq_n], dtype=torch.long)[:, 0]

        return state_seq, action

    def __getitem__(self, idx):
        state_seq, action = self.get_state_action(idx)
        next_state_seq = torch.zeros(len(state_seq), dtype=torch.long)
        next_state_seq[:-1] = state_seq[1:]
        next_state_seq[-1] = action

        next_state_seq = nn.functional.pad(next_state_seq, (self.n - len(next_state_seq), 0), mode='constant', value=self.pad_token)
        # state_seq = nn.functional.pad(state_seq, (self.n - len(state_seq), 0), mode='constant', value=self.pad_token)
        # pi_e_s = torch.softmax(self.pi_e.score(state_seq), 1).squeeze(0)

        reward = self.rewards[idx]

        actions_neg = torch.tensor(np.random.choice(self.all_actions[~np.isin(self.all_actions, state_seq)],
                                                    self.n_neg_samples,
                                                    replace=False), dtype=torch.long)
        state = torch.from_numpy(self.states[idx]).float()
        next_state = torch.from_numpy(self.next_states[idx]).float()

        # assert pi_e_s.requires_grad == False

        return reward, state, next_state, next_state_seq, action, actions_neg

class RLDatasetOnlineVal(RLDatasetOnline):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.full_sequences)

    def get_seq(self, idx):
        if self.n < 0:
            seq = torch.tensor(self.full_sequences[idx][1][:-1], dtype=torch.long)
        else:
            seq = self.full_sequences[idx][1]
            if len(seq) < self.n + 1:
                seq = np.pad(seq, (self.n - len(seq), 0), mode='constant', constant_values=self.pad_token)

            seq = torch.tensor(seq[-self.n-1:-1], dtype=torch.long)

        return seq


class FQE:
    def __init__(self,
                 dataset,
                 val_dataset,
                 pi_e,
                 optim_conf,
                 n_epochs,
                 state_dim,
                 n_actions,
                 hidden_size,
                 gamma,
                 device):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.pi_e = pi_e
        self.optim_conf = optim_conf
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.device = device
        # self.device = "cpu"

        # self.pi_e = self.pi_e.to(self.device)
        # self.pi_e.eval()

        self.q = QNetwork(self.state_dim,
                          self.n_actions,
                          self.hidden_size).to(self.device)

    @staticmethod
    def copy_over_to(source, target):
        target.load_state_dict(source.state_dict())

    def train(self, batch_size, plot_info=True):
        optimizer = optim.Adam(self.q.parameters(), **self.optim_conf)
        q_prev = QNetwork(self.state_dim,
                          self.n_actions,
                          self.hidden_size).to(self.device)

        self.copy_over_to(self.q, q_prev)

        val_idxes = np.random.choice(np.arange(len(self.val_dataset)), 200)
        val_states_seq = [self.val_dataset.get_seq(idx) for idx in val_idxes]
        val_states = [torch.from_numpy(self.val_dataset.states[idx]).float() for idx in val_idxes]

        values = []
        for epoch in range(self.n_epochs):
            dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=6)

            loss_history = []
            for rewards, states, next_states, next_state_seqs, actions, actions_neg in tqdm(dataloader, total=len(dataloader)):
                rewards = rewards.to(self.device)
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                next_state_seqs = next_state_seqs.to(self.device)
                actions = actions.to(self.device)
                actions_neg = actions_neg.to(self.device)

                next_state_seqs = next_state_seqs[:, -self.pi_e.num_embeddings:]

                with torch.no_grad():
                    # states = self.pi_e.log2feats(state_seqs)[:, -1]
                    #CHANGED:
                    #------------------------------------------------------------
                    # next_states_e = self.pi_e.log2feats(next_state_seqs)[:, -1]

                    # item_embs = self.pi_e.item_emb.weight
                    # logits = item_embs.matmul(next_states_e.unsqueeze(-1)).squeeze(-1)
                    #------------------------------------------------------------
                    logits = self.pi_e.score_batch(next_state_seqs)
                    pi_e_s = torch.softmax(logits, 1)

                    q_vals = q_prev(next_states)

                    q_vals = (pi_e_s * q_vals).sum(axis=-1)

                    y = rewards + self.gamma * q_vals
                    y_neg = self.gamma * q_vals # (batch_size)

                preds = self.predict(states, actions.unsqueeze(-1)).squeeze(-1) # (batch_size)
                preds_neg = self.predict(states, actions_neg) # (batch_size, n_neg_samples)

                assert len(preds.shape) == 1
                assert len(y.shape) == 1

                loss = torch.sum((preds - y)**2) + torch.sum((preds_neg - y_neg.unsqueeze(-1))**2)
                loss = loss / (preds.numel() + preds_neg.numel())
                optimizer.zero_grad()
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
                optimizer.step()

                loss_history.append(loss.item())

            val_probs = [torch.softmax(self.pi_e.score(state_seq.to(self.device)), 1).squeeze(0).detach().cpu() for state_seq in val_states_seq]
            val_actions = [np.random.choice(np.arange(self.n_actions), p=prob.numpy()) for prob in val_probs]
            val_preds = self.predict(torch.stack(val_states, dim=0).to(self.device),
                                     torch.tensor(val_actions, device=self.device, dtype=torch.long).unsqueeze(-1)).detach()

            values.append(torch.mean(val_preds).item())

            if plot_info:
                self.plot_info(loss_history, values)

            print(f"Last iter loss = {loss_history[-1]}, value on val = {values[-1]}")

            self.copy_over_to(self.q, q_prev)

            print(f"Finished Epoch {epoch}.")


        return values

    def predict(self, states, actions):
        return torch.take_along_dim(self.q(states), actions, dim=1)

    def plot_info(self, loss_history, values):
        fig = plt.figure(figsize=(20, 10))

        fig.add_subplot(1, 2, 1)
        plt.plot(loss_history[::10])
        plt.yscale("log")
        plt.grid(True)

        fig.add_subplot(1, 2, 2)
        plt.plot(values)
        plt.grid(True)

        plt.savefig("plot.png")
        plt.show()














