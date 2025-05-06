# DFD: Data-privacy-regulation Food Delivery

**Article:** 
Zijian Zhao, Sen Li\*, "The Impacts of Data Privacy Regulations on Food-Delivery Platforms" (under review)
Zijian Zhao, Sen Li\*, "Discriminatory Order Assignment and Payment-Setting on Food-Delivery Platforms: A Multi-Action and Multi-Agent Reinforcement Learning Framework" (under review)

**Previous Repository:**
[RS2002/Double-PDF: Official Repository for The Paper, Discriminatory Order Assignment and Payment-Setting on Food-Delivery Platforms: A Multi-Action and Multi-Agent Reinforcement Learning Framework](https://github.com/RS2002/Double-PDF)

**Notice:**
Please note there is a bug in detour time calculation and we will fix it later.



## 1. Workflow

![](./img/workflow.png)

![](./img/main.png)

![](./img/network.png)

## 2. Dataset

Please refer our previous work [RS2002/Double-PDF: Official Repository for The Paper, Discriminatory Order Assignment and Payment-Setting on Food-Delivery Platforms: A Multi-Action and Multi-Agent Reinforcement Learning Framework](https://github.com/RS2002/Double-PDF).



## 3. How to Run

### 3.1 Platform Training

#### 3.1.1 Pre-training

```shell
python pretrain_platform.py --probability_worker --rand_sample_rate
```



#### 3.1.2 Fine-tuning

```shell
python finetune_platform.py --probability_worker --demand_sample_rate <order sampling rate> --worker_mode <gdpr/benchmark/mask>
```



### 3.2 Courier Training

```shell
python train_courier.py --probability_worker --model_path <platform_strategy_model_path> --worker_mode <gdpr/benchmark/mask> --lowest_utility <minimum_wage_for_exit_couriers> --demand_sample_rate <order sampling rate>
```



### 3.3 Comparative Evaluation

```shell
python eval.py --probability_worker --gdpr_path <gdpr_model_path> --benchmark_path <benchmark_model_path> --mask_path <mask_model_path> --lowest_utility <minimum_wage_for_exit_couriers> --demand_sample_rate <order sampling rate>
```



## 4. Reference

```

```

