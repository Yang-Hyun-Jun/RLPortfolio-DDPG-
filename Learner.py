import torch
import Visualizer
import numpy as np

from Environment import environment
from Agent import agent
from ReplayMemory import ReplayMemory
from Noise import OUProcess
from Noise import Normal
from Network import Actor
from Network import Critic
from Metrics import Metrics

class learner:

    def __init__(self,
                 tau=0.005, delta=0.05,
                 discount_factor=0.90,
                 batch_size=30, memory_size=100,
                 chart_data=None, K=None, lr=1e-4,
                 min_trading_price=None, max_trading_price=None):

        assert min_trading_price >= 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price

        self.environment = environment(chart_data)
        self.memory = ReplayMemory(max_size=memory_size)
        self.chart_data = chart_data
        self.batch_size = batch_size

        self.actor = Actor(K=K)
        self.actor_target = Actor(K=K)
        self.critic = Critic(K=K)
        self.critic_target = Critic(K=K)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.ou_noise = OUProcess(np.zeros(K))
        self.normal_noise = Normal(mu=0, std=0.1, size=K)

        self.lr = lr
        self.tau = tau
        self.K = K
        self.delta = delta
        self.discount_factor = discount_factor
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        self.agent = agent(environment=self.environment,
                           critic=self.critic,
                           critic_target=self.critic_target,
                           actor=self.actor,
                           actor_target=self.actor_target,
                           lr=self.lr, K=self.K,
                           tau=self.tau, delta=self.delta,
                           discount_factor=self.discount_factor,
                           min_trading_price=min_trading_price,
                           max_trading_price=max_trading_price)

    def reset(self):
        self.environment.reset()
        self.agent.reset()

    @staticmethod
    def prepare_training_inputs(sampled_exps):
        states1 = []
        states2 = []
        actions = []
        rewards = []
        next_states1 = []
        next_states2 = []
        dones = []

        for sampled_exp in sampled_exps:
            states1.append(sampled_exp[0])
            states2.append(sampled_exp[1])
            actions.append(sampled_exp[2])
            rewards.append(sampled_exp[3])
            next_states1.append(sampled_exp[4])
            next_states2.append(sampled_exp[5])
            dones.append(sampled_exp[6])

        states1 = torch.cat(states1, dim=0).float()
        states2 = torch.cat(states2, dim=0).float()
        actions = torch.cat(actions, dim=0).float()
        rewards = torch.cat(rewards, dim=0).float()
        next_states1 = torch.cat(next_states1, dim=0).float()
        next_states2 = torch.cat(next_states2, dim=0).float()
        dones = torch.cat(dones, dim=0).float()
        return states1, states2, actions, rewards, next_states1, next_states2, dones


    def run(self, num_episode=None, balance=None):
        self.agent.set_balance(balance)
        metrics = Metrics()
        steps_done = 0

        for episode in range(num_episode):
            self.reset()
            cum_r = 0
            state1 = self.environment.observe()
            portfolio = self.agent.portfolio
            while True:
                action_, confidence = self.agent.get_action(torch.tensor(state1).float().view(1, self.K, -1),
                                                            torch.tensor(portfolio).float().view(1, self.K+1, -1))

                ou_noise = self.ou_noise()
                no_noise = self.normal_noise()
                # action_ = action_ + no_noise
                action = action_.clip(-1.0, 1.0)
                next_state1, next_portfolio, reward, done = self.agent.step(action, confidence)
                steps_done += 1

                experience = (torch.tensor(state1).float().view(1, self.K, -1),
                              torch.tensor(portfolio).float().view(1, self.K+1, -1),
                              torch.tensor(action).float().view(1, -1),
                              torch.tensor(reward).float().view(1,-1),
                              torch.tensor(next_state1).float().view(1, self.K, -1),
                              torch.tensor(next_portfolio).float().view(1, self.K+1, -1),
                              torch.tensor(done).float().view(1,-1))

                self.memory.push(experience)
                cum_r += reward
                state1 = next_state1
                portfolio = next_portfolio

                if steps_done % 300 == 0:
                    q_value = self.agent.critic(torch.tensor(state1).float().view(1, self.K, -1),
                                                torch.tensor(portfolio).float().view(1, self.K+1, -1),
                                                torch.tensor(action).float().view(1, -1)).detach().numpy()[0]

                    a = action
                    p = self.agent.portfolio
                    pv = self.agent.portfolio_value
                    sv = self.agent.portfolio_value_static
                    balance = self.agent.balance
                    change = self.agent.change
                    pi_vector = self.agent.pi_operator(change)
                    profitloss = self.agent.profitloss
                    np.set_printoptions(precision=4, suppress=True)
                    print(f"episode:{episode} ------------------------------------------------------------------------")
                    print(f"price:{self.environment.get_price()}")
                    print(f"q_value:{q_value}")
                    print(f"noise:{no_noise}")
                    print(f"action_:{action_}")
                    print(f"action:{a}")
                    print(f"portfolio:{p}")
                    print(f"pi_vector:{pi_vector}")
                    print(f"portfolio value:{pv}")
                    print(f"static value:{sv}")
                    print(f"balance:{balance}")
                    print(f"cum reward:{cum_r}")
                    print(f"profitloss:{profitloss}")
                    print(f"actor_loss:{self.agent.actor_loss}")
                    print(f"critic_loss:{self.agent.critic_loss}")
                    print("-------------------------------------------------------------------------------------------")


                #학습
                if len(self.memory) >= self.batch_size:
                    sampled_exps = self.memory.sample(self.batch_size)
                    sampled_exps = self.prepare_training_inputs(sampled_exps)
                    self.agent.update(*sampled_exps)
                    self.agent.soft_target_update(self.agent.critic.parameters(), self.agent.critic_target.parameters())
                    self.agent.soft_target_update(self.agent.actor.parameters(), self.agent.actor_target.parameters())

                #metrics 마지막 episode 대해서만
                if episode == range(num_episode)[-1]:
                    metrics.portfolio_values.append(self.agent.portfolio_value)
                    metrics.profitlosses.append(self.agent.profitloss)

                if done:
                    break

            if episode == range(num_episode)[-1]:
                #metric 계산과 저장
                metrics.get_profitlosses()
                metrics.get_portfolio_values()

                #계산한 metric 시각화와 저장
                Visualizer.get_portfolio_value_curve(metrics.portfolio_values)
                Visualizer.get_profitloss_curve(metrics.profitlosses)

    def save_model(self, critic_path, actor_path):
        torch.save(self.agent.critic.state_dict(), critic_path)
        torch.save(self.agent.actor.state_dict(), actor_path)