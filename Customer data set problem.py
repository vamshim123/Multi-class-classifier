
# coding: utf-8

# In[1]:


import pandas as pd


# Get the datset from here =======> https://catalog.data.gov/dataset/consumer-complaint-database

# In[4]:


df=pd.read_csv("Consumer_Complaints.csv")


# In[5]:


len(df)


# In[6]:


df.drop(['Company response to consumer'],axis=1,inplace=True)


# In[7]:


df.drop(['Date sent to company'],axis=1,inplace=True)


# In[8]:


df.head()


# Now lets clean up the data 
#             as we are predicting "product"---output by giving "Consumer_complaint_narrative" as input lets drop() all the remaining colomns

# In[9]:


from io import StringIO


# In[10]:


col = ['Product', 'Consumer complaint narrative']
df = df[col]


# In[11]:


df = df[pd.notnull(df['Consumer complaint narrative'])]
df.columns = ['Product', 'Consumer_complaint_narrative']


# In[12]:


df['category_id'] = df['Product'].factorize()[0]
category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)
df.head()


# By using %matplotlib lets see the number of complaints per product

# In[14]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Product').Consumer_complaint_narrative.count().plot.bar(ylim=0)
plt.show


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')


# In[18]:


features = tfidf.fit_transform(df.Consumer_complaint_narrative)


# In[19]:


labels = df.category_id


# In[22]:


features.shape


# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# clf = MultinomialNB().fit(X_train_tfidf, y_train)

# fitting the data and Training the model

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[24]:


from sklearn.naive_bayes import MultinomialNB


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)


# In[26]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)


# In[27]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[28]:


clf = MultinomialNB().fit(X_train_tfidf, y_train)


# Lets see some predictions 

# In[32]:


pred=clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."]))


# In[37]:


print(pred)


# In[30]:


print(clf.predict(count_vect.transform(["I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"])))


# we can use other classifiers like liner SVC and logical regression.
