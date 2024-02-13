import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

dataset=pd.read_excel('C:\OS\Machine Learning Lab\LAB3-13.02.24\irctc.xlsx')

a=dataset.head()
print(a)

mean = statistics.mean(dataset['Price'])
variance = statistics.variance(dataset['Price'])


if 'Wednesday' in dataset['Day'].values:
    wednesday_prices = dataset[dataset['Day'] == 'Wednesday']['Price']
    wednesday_mean = statistics.mean(wednesday_prices)
    print("Mean price for Wednesdays:", wednesday_mean)
else:
    print("No data available for Wednesdays.")


if 'Apr' in data['Month'].values:
    april_prices = data[data['Month'] == 'Apr']['Price']
    april_mean = statistics.mean(april_prices)
    print("Mean price for April:", april_mean)
else:
    print("No data available for April.")


loss_probability = len(dataset[dataset['Chg%'] < 0]) / len(dataset['Chg%'])
print("Probability of making a loss over the stock:", loss_probability)

wednesday_profit_probability = len(dataset[(dataset['Day'] == 'Wednesday') & (dataset['Chg%'] > 0)]) / len(wednesday_prices)
print("Probability of making a profit on Wednesday:", wednesday_profit_probability)


conditional_probability = wednesday_profit_probability / (len(dataset[dataset['Day'] == 'Wednesday']) / len(dataset))
print("Conditional probability of making profit, given that today is Wednesday:", conditional_probability)

plt.scatter(dataset['Day'], dataset['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Chg% vs Day of the Week')
plt.show()