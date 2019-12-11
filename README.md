# NGBoost vs. LightGBM vs. XGBoost

It could be painful for some OS to build XGBoost or LightGBM from the source. So here we containerize the env to make our life easier 🙂



## How to reproduce the result

```bash
# Result:
+----------+-----------+------------+--------+---------------------+
|   Name   | Iteration | Estimators |  RMSE  |         Time        |
+----------+-----------+------------+--------+---------------------+
| NGBoost  |     NA    |    1000    | 0.1203 |   6.07675017695874  |
| NGBoost  |     NA    |    5000    | 0.1142 |  30.040924307890236 |
| LightGBM |    1000   |     NA     | 0.1523 | 0.14832178968936205 |
| LightGBM |    5000   |     NA     | 0.1171 |  0.6903378637507558 |
| XGBoost  |    1000   |     NA     | 0.1468 |   1.19044853374362  |
| XGBoost  |    5000   |     NA     | 0.1314 |  6.730385942384601  |
+----------+-----------+------------+--------+---------------------+
All Done: 100%|████████████████████████████████████████████████████████████████████████████████| 8/8 [01:08<00:00,  9.85s/it]
```

1. Build from scratch

```bash
$ git pull https://github.com/benbenbang:ngboost-explore
$ docker-compose build
$ docker-compose run ngboost-explore
```

2. Pull from Dockerhub

```bash
# Note that since the data is already packed insided the container, it takes around 1.1 GB
$ docker pull benbenbang/ngboost-explore:latest
$ docker run ngboost-explore
```

3. Clone from Githhub

```bash
$ git clone https://github.com/benbenbang/ngboost-explore.git
$ cd ngboost-explore

# Don't forget to activate your test environment first
$ pip install -r requirements.txt
$ pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git

# You will need two environment variables before you run the calcu
$ export SEED=5566
$ export NUM_PLOTS=4
$ python src/benchmark.py
```

ps. 
- To get different result, simply change the seed in docker-compose file
- If you want to reuse the environment after tweeking the hyperparameters, please use method 1 or 3 instead of using a static copy file in method 2.
- To reproduce the probabiliy distribution plot, please use method 1 or 3 to rerun the process. Here's a sample:
  <img src="https://d.pr/i/5ijtf5.png" alt="prob_dist" style="zoom:70%;" />

