# Amazon_BuyBox_Reinforcement_Learning
 Multi-agent price competition in Amazon Buy Box environment. This project aims to answer the following questions 

1. How does the Amazon marketplace with the Buy Box (or Amazon Featured Offer) mechanism differ from the classic Bertrand competition scenario in Industrial Organization?
2. Given the exisiting Repricer services in the market, can we observe evidence of supracompetitive equilibrium prices (i.e. as described in Asker et al. 2022)?
3. Given the heterogeneity of sellers (e.g. size, history, comments), what are the best dynamic pricing strategies for the sellers to win the Buy Box and win over sales in each ASIN market?

To answer these questions:

1. We use deep learning models to learn and mimic the Buy Box assignment mechanism given the state of the market. This step learns the model from observational data from Amazon.
2. Given the estimated model, we simulate multi-agent scenarios where each agent can choose the specific pricing rules (i.e. match the buy box price, match the lowest price, undercut Amazon's price etc.) in each period. Each agent also has heterogeneity in their marginal cost, and whether they are fulfilled by Amazon. (FBA)
3. We assume there is a simple demand-side model due to the lack of observed transation data.
4. We observe and report the scenarios most suitable for price collusion, and the best strategy for different sellers to survive longer and win the buy box.
