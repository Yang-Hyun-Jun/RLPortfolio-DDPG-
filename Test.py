import DataManager
import Visualizer
import utils
import torch
import numpy as np
from Metrics import Metrics
from Environment import environment
from Agent import agent
from Network import Actor
from Network import Critic
from Network import Score

if __name__ == "__main__":
    stock_code = ["010140", "000810", "034220"]

    path_list = []
    for code in stock_code:
        path = utils.Base_DIR + "/" + code
        path_list.append(path)

    #test data load
    train_data, test_data = DataManager.get_data_tensor(path_list,
                                                        train_date_start="20090101",
                                                        train_date_end="20150101",
                                                        test_date_start="20170102",
                                                        test_date_end=None)

    #dimension
    state1_dim = 5
    state2_dim = 2
    K = 3

    #Test Model load
    # score_net_actor = Score()
    # score_net_critic = Score()
    # actor = Actor(score_net_actor)
    # actor_target = Actor(score_net_actor)
    # critic = Critic(score_net_critic)
    # critic_target = Critic(score_net_critic)

    actor = Actor()
    actor_target = Actor()
    critic = Critic()
    critic_target = Critic()

    balance = 15000000
    min_trading_price = 0
    max_trading_price = 5000000

    #Agent
    environment = environment(chart_data=test_data)
    agent = agent(environment=environment,
                  actor=actor,
                  actor_target=actor_target,
                  critic=critic,
                  critic_target=critic_target,
                  critic_lr=1e-3, actor_lr=1e-3,
                  tau=0.005, discount_factor=0.9, delta=0.0,
                  min_trading_price=min_trading_price,
                  max_trading_price=max_trading_price)

    #Model parameter load
    critic_path = utils.SAVE_DIR + "/Models" + "/DDPGPortfolio_critic.pth"
    actor_path = utils.SAVE_DIR + "/Models" + "/DDPGPortfolio_actor.pth"
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))

    #Test
    metrics = Metrics()
    agent.set_balance(balance)
    agent.reset()
    agent.environment.reset()
    agent.epsilon = 0
    state1 = agent.environment.observe()
    portfolio = agent.portfolio
    steps_done = 0

    while True:
        action, confidence = agent.get_action(torch.tensor(state1).float().view(1, K, -1),
                                              torch.tensor(portfolio).float().view(1, K+1, -1))

        next_state1, next_portfolio, reward, done = agent.step(action, confidence)

        steps_done += 1
        state1 = next_state1
        portfolio = next_portfolio

        metrics.portfolio_values.append(agent.portfolio_value)
        metrics.profitlosses.append(agent.profitloss)

        if steps_done % 50 == 0:
            a = action
            p = agent.portfolio
            pv = agent.portfolio_value
            sv = agent.portfolio_value_static
            b = agent.balance
            change = agent.change
            pi_vector = agent.pi_operator(change)
            profitloss = agent.profitloss
            np.set_printoptions(precision=4, suppress=True)
            # print("------------------------------------------------------------------------------------------")
            # print(f"price:{environment.get_price()}")
            # print(f"q_value:{q_value}")
            # print(f"action:{a}")
            # print(f"portfolio:{p}")
            # print(f"pi_vector:{pi_vector}")
            # print(f"portfolio value:{pv}")
            # print(f"static value:{sv}")
            print(f"balance:{b}")
            # print(f"profitloss:{profitloss}")
            # print("-------------------------------------------------------------------------------------------")

        if done:
            break

    bench_profitloss1 = []
    agent.set_balance(15000000)
    agent.reset()
    agent.environment.reset()
    state1 = agent.environment.observe()
    portfolio = agent.portfolio
    steps_done = 0
    while True:
        steps_done += 1
        action = np.array([0.33, 0.33, 0.33])

        confidence = abs(action)
        next_state1, next_portfolio, reward, done = agent.step(action, confidence)
        steps_done += 1

        state1 = next_state1
        portfolio = next_portfolio
        bench_profitloss1.append(agent.profitloss)
        if done:
            break


    bench_profitloss2 = []
    agent.set_balance(15000000)
    agent.reset()
    agent.environment.reset()
    state1 = agent.environment.observe()
    portfolio = agent.portfolio
    while True:
        action = np.random.uniform(low=-0.1, high=0.1, size=3)
        confidence = abs(action)
        next_state1, next_portfolio, reward, done = agent.step(action, confidence)
        steps_done += 1

        state1 = next_state1
        portfolio = next_portfolio
        bench_profitloss2.append(agent.profitloss)
        if done:
            break

    Vsave_path2 = utils.SAVE_DIR + "/" + "/Metrics" + "/Portfolio Value Curve_test"
    Vsave_path4 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss Curve_test"
    Msave_path1 = utils.SAVE_DIR + "/" + "/Metrics" + "/Portfolio Value_test"
    Msave_path3 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss_test"

    metrics.get_portfolio_values(save_path=Msave_path1)
    metrics.get_profitlosses(save_path=Msave_path3)

    Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
    Visualizer.get_profitloss_curve(metrics.profitlosses, bench_profitloss1, save_path=Vsave_path4)

