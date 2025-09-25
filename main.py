import torch


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        pass

    def cache(self, experience):
        """Add the experience to memory"""
        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        pass


if __name__ == "__main__":
    mario = Mario()
