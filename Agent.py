import torch
import torch.nn as nn
import numpy as np

class agent(nn.Module):
    # TRADING_CHARGE = 0.00015
    # TRADING_TEX = 0.0025
    TRADING_CHARGE = 0.0
    TRADING_TEX = 0.0

    ACTIONS = []
    NUM_ASSETS = 0
    NUM_ACTIONS = 0

    def __init__(self, environment,
                 critic:nn.Module,
                 critic_target:nn.Module,
                 actor:nn.Module,
                 actor_target:nn.Module,
                 lr:float, K:int,
                 tau:float, delta:float,
                 discount_factor:float,
                 min_trading_price:int,
                 max_trading_price:int):

        super().__init__()
        self.environment = environment
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        self.critic = critic
        self.critic_target = critic_target
        self.actor = actor
        self.actor_target = actor_target
        self.lr = lr
        self.tau = tau
        self.K = K
        self.delta = delta
        self.discount_factor = discount_factor

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=self.lr)
        self.huber = nn.SmoothL1Loss()

        self.critic.load_state_dict(self.critic_target.state_dict())
        self.actor.load_state_dict(self.actor_target.state_dict())

        self.num_stocks = np.array([0] * self.K)
        self.portfolio = np.array([0] * (self.K+1), dtype=float)

        self.portfolio_value = 0
        self.initial_balance = 0
        self.balance = 0
        self.profitloss = 0


    def set_balance(self, balance):
        self.initial_balance = balance

    def reset(self):
        self.num_stocks = np.array([0] * self.K)
        self.portfolio = np.array([0] * (self.K+1), dtype=float)
        self.portfolio[0] = 1
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.profitloss = 0

    def get_action(self, state1, portfolio):
        with torch.no_grad():
            action = self.actor(state1, portfolio)[0].numpy()
            confidence = abs(action)
        return action, confidence

    def decide_trading_unit(self, confidence, price):
        trading_price = self.portfolio_value * confidence
        trading_unit = int(np.array(trading_price)/price)
        return trading_unit

    def validate_action(self, actions, delta):
        for i in range(actions.shape[0]):
            if delta < actions[i] <= 1:
                # 매수인 경우 적어도 1주를 살 수 있는지 확인
                if self.balance < self.environment.get_price()[i] * (1 + self.TRADING_CHARGE):
                    actions[i] = 0.0 #Hold

            elif -1 <= actions[i] < -delta:
                # 매도인 경우 주식 잔고가 있는지 확인
                if self.num_stocks[i] == 0:
                    actions[i] = 0.0 #Hold

    def pi_operator(self, change_rate):
        pi_vector = np.zeros(len(change_rate) + 1)
        pi_vector[0] = 1
        pi_vector[1:] = change_rate + 1
        return pi_vector

    def get_portfolio_value(self, close_p1, close_p2, portfolio):
        close_p2 = np.array(close_p2)
        close_p1 = np.array(close_p1)
        change_rate = (close_p2 - close_p1)/close_p1
        pi_vector = self.pi_operator(change_rate)
        portfolio_value = np.dot(self.portfolio_value * portfolio, pi_vector)
        return portfolio_value

    def get_portfolio(self, close_p1, close_p2):
        close_p2 = np.array(close_p2)
        close_p1 = np.array(close_p1)
        change_rate = (close_p2 - close_p1)/close_p1
        pi_vector = self.pi_operator(change_rate)
        portfolio = (self.portfolio * pi_vector)/(np.dot(self.portfolio, pi_vector))
        return portfolio

    def get_reward(self, pv, pv_static):
        reward = (pv-pv_static)/pv_static
        return reward

    def step(self, action, confidence):
        assert action.shape[0] == confidence.shape[0]
        assert 0 <= self.delta < 1

        close_p1 = self.environment.get_price()

        self.validate_action(action, self.delta)
        self.portfolio_value_static_ = self.portfolio * self.portfolio_value

        #우선 전체적으로 종목별 매도 수행을 먼저한다.
        for i in range(action.shape[0]):
            p1_price = close_p1[i]

            if abs(action[i]) > 1.0:
                raise Exception("Action is out of bound")
            # Sell
            if -1 <= action[i] < -self.delta:
                trading_unit = self.decide_trading_unit(confidence[i], p1_price)
                trading_unit = min(trading_unit, self.num_stocks[i])

                invest_amount = p1_price * (1 - (self.TRADING_TEX + self.TRADING_CHARGE)) * trading_unit
                self.num_stocks[i] -= trading_unit
                self.balance += invest_amount
                self.portfolio[0] += invest_amount/self.portfolio_value
                self.portfolio[i+1] -= invest_amount/self.portfolio_value


        #다음으로 종목별 매수 수행
        for i in range(action.shape[0]):
            p1_price = close_p1[i]

            if abs(action[i]) > 1.0:
                raise Exception("Action is out of bound")
            # Buy
            if self.delta < action[i] <= 1:
                trading_unit = self.decide_trading_unit(confidence[i], p1_price)
                cal_balance = (self.balance - p1_price * (1 + self.TRADING_CHARGE) * trading_unit)

                #돈 부족 한 경우
                if cal_balance < 0:
                    trading_unit = min(
                        int(self.balance / (p1_price * (1 + self.TRADING_CHARGE))),
                        int(self.max_trading_price / p1_price))

                # 수수료 적용하여 총 매수 금액 산정
                invest_amount = p1_price * (1 + self.TRADING_CHARGE) * trading_unit
                self.num_stocks[i] += trading_unit
                self.balance -= invest_amount
                self.portfolio[0] -= invest_amount/self.portfolio_value
                self.portfolio[i+1] += invest_amount/self.portfolio_value

        self.portfolio = self.portfolio / np.sum(self.portfolio) #sum = 1

        #다음 상태로 진행
        next_state1 = self.environment.observe()
        next_portfolio = self.portfolio
        close_p2 = self.environment.get_price()

        self.change = (np.array(close_p2)-np.array(close_p1))/np.array(close_p1)
        self.portfolio = self.get_portfolio(close_p1=close_p1, close_p2=close_p2)
        self.portfolio_value = self.get_portfolio_value(close_p1=close_p1, close_p2=close_p2, portfolio=self.portfolio)
        self.portfolio_value_static = np.dot(self.portfolio_value_static_, self.pi_operator(self.change))
        self.profitloss = ((self.portfolio_value / self.initial_balance) - 1)*100

        reward = self.get_reward(self.portfolio_value, self.portfolio_value_static)
        reward = reward*100

        if len(self.environment.chart_data)-1 <= self.environment.idx:
            done = 1
        else:
            done = 0
        return next_state1, next_portfolio, reward, done

    def update(self, s_tensor, portfolio, action, reward, ns_tensor, ns_portfolio, done):

        with torch.no_grad():
            next_action = self.actor_target(ns_tensor, ns_portfolio)
            next_value = self.critic_target(ns_tensor, ns_portfolio, next_action)
            target = reward + self.discount_factor * next_value * (1-done)

        value = self.critic(s_tensor, portfolio, action)
        self.critic_loss = self.huber(value, target)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_loss = -self.critic(s_tensor, portfolio, self.actor(s_tensor, portfolio)).mean()
        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()

    def soft_target_update(self, params, target_params):
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def hard_target_update(self):
        self.critic.load_state_dict(self.critic_target.state_dict())
        self.actor.load_state_dict(self.actor_target.state_dict())


