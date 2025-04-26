#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   AmazonLogitDemandPricing_env.py
@Time    :   2025/04/26 14:33:16
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   A two-firm repeated pricing environment.
    Actions: tuple (a_i, a_j) with each in {0,...,M-1}, indexing price grid.
    Observation: tuple (p_i_lag, p_j_lag) with each in {0,...,M-1}, indexing price grid. Price pair played in last step.
    Rewards: profits for each firm given marginal cost and demand rule.
    Demand: Logit demand with outside option. The Amazon Featured Offer (or Buy Box) winner has an additional utility term in the demand function.
    The Buy Box is assigned to one of the sellers base on its features using a prediction model trained on real-world Amazon Buy Box assignment data.
    More about the prediction model:
        1. We obtained the historical pricingd data for the top 5000 best sellers in the Books category on Amazon. For each ASIN, we collecte the pricing history data for all the sellers, their offer/seller feature, and the Buy Box winner at each time step.
        2. We trained a prediction model to predict the Buy Box winner based on the seller features and the prices of all the sellers.
        3. We used the prediction model to generate the additional utility term for the Buy Box winner in the demand function. The prediction result is also stored in a tabular to avoid repeated computation.
'''
