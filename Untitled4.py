#!/usr/bin/env python
# coding: utf-8

# # GRIP (The Sparks Foundation)
# 
# ## Data Science & Business Analytics Internship 
# 
# ### Task 1: Prediction using Supervised ML
# 
# ## Author: Abhinav Raj

# In[2]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


# Reading data from remote link 
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df=pd.read_csv(url)
print("Data Imported Successfully")
df


# In[7]:


df.describe()


# In[8]:


# Plotting the distribution of scores

get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.scatter(df.Hours, df.Scores, color = 'red', marker = '+')


# ###### From the graph, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score

# ### Preparing the Data 
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
# 

# In[9]:


x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# ##### Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit.learn,s model built-in train_test_split() method

# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# ### Training the Algorithm 
# 
# ##### We have split our data into training and test sets, and now is finally the time to train our algorithm

# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("Training Complete")


# In[12]:


# Plotting the Regression line
line = regressor.coef_*x+regressor.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line)
plt.show()


# ### Making Predictions 
# 
# ##### Now that we have trained our algorithm, its time to make some predictions.

# In[13]:


print(x_test)
y_pred = regressor.predict(x_test) # Predicting the scores


# In[14]:


# Comparing Actual vs Predicted 
df = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
df


# ### What will be the predicted score if a student studies for 9.25 hrs/day

# In[15]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print("No. of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### Evaluating the model 
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics. 

# In[16]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




