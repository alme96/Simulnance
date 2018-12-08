Group Name: Simulnance

Group participants names: Kedir Firehiwot Nesro, Menichelli Alessandro, Wüs Vincent

Project Title: Trade!t

Programming language: Python

Reproducibility

In our model we want to simulate a financial market.
For this purpose we used a agent based approach.
The agents represent traders, which trade one single asset.
To simulate how traders place their offers we had to make assumptions on how they behave.
In the order generation process we assume that traders place their offers with respect to a normal distribution.

In this simple reproducibility test we let you simulate the market and plot the estimated price path of the stock.

1. Make sure to have all the packages needed installed. The two packages "matplotlib" and "mesa" are not preinstalled.
2. Run the file "Reproducibility_light" with python without changing any of the code. After say 5 seconds you should receive a plot of the price path. This result is obtained for a small standard deviation of the normal distribution, which corresponds to traders behaving all very simmilar.
3. Search for the global variable "std" which has been set to 0.001 by default. Change it to 0.01 and run the programm again.
The plot which you receive this time should vary stronger around 100 then before, which corresponds to traders behaving very differently. This happens when there's high uncertainty about te actuall stock price.


