# GFD: GDPR Food Delivery

**Article:** Zijian Zhao, Sen Li*, "The Impact of GDPR on On-Demand Food-Delivery Platforms with Discriminatory Order Assignment and Personalized Courier Payment" (under way)



**Notice:**

1. Please note there is a bug in detour time calculation and we will fix it later.
2. We have improved the model according to our latest work [Triple-BERT](https://github.com/RS2002/Triple-BERT). The original version is in the `ori` folder.
3. The code is based on our previous work [RS2002/Double-PDF: Official Repository for The Paper, Discriminatory Order Assignment and Payment-Setting on Food-Delivery Platforms: A Multi-Action and Multi-Agent Reinforcement Learning Framework](https://github.com/RS2002/Double-PDF).



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

