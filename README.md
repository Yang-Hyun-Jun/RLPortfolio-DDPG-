# DDPG Portfolio management 

 - 논문 작성, 실험을 위한 비교 모델
 - DDPG style update

## Action Define

 1. (-1, 1)  사이의 값을 action으로 정의
 2. **Portfolio vector를 action으로 정의** 

2의 방식에 따라 $trading_{t}=(Desired PF-PF_{t-1})[1:]$  
2의 방식으로 trading unit scale을 안정화 

## Training 

 - DDPG model training cumulative portfolio value 19876427 won
 - DDPG model training cumulative profitloss 32%
 
![Portfolio value curve train](Images/Portfolio%20Value%20Curve_train.png)

## Test

 - DDPG model test cumulative profitloss 32%
 - B&H benchmarking test cumulative profitloss 22%
 
![Portfolio value curve test](Images/Portfolio%20Value%20Curve_test.png)
![Portfoliio profitloss curve test](Images/Profitloss%20Curve_test.png)
