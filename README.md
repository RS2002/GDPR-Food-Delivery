# GFD: GDPR Food Delivery

**Article:**

The code is based on [RS2002/Double-PDF: Official Repository for The Paper, Discriminatory Order Assignment and Payment-Setting on Food-Delivery Platforms: A Multi-Action and Multi-Agent Reinforcement Learning Framework](https://github.com/RS2002/Double-PDF).



## 1. Workflow

![](./img/workflow.png)

![](./img/main.png)

![](./img/network.png)

## 2. Dataset

Please refer our previous work [RS2002/Double-PDF: Official Repository for The Paper, Discriminatory Order Assignment and Payment-Setting on Food-Delivery Platforms: A Multi-Action and Multi-Agent Reinforcement Learning Framework](https://github.com/RS2002/Double-PDF).



## 3. How to Run

### 3.1 Platform Training

```shell
python pretrain.py --probability_worker
```



### 3.2 Courier Training

```shell
python train_courier --probability_worker --model_path <platform_strategy_model_path> --worker_mode <gdpr/benchmark/mask> --lowest_utility <minimum_wage_for_exit_couriers>
```



### 3.3 Evaluation

```shell
python eval.py --probability_worker --model_path <courier_strategy_model_path>
```



## 4. Reference

```

```

