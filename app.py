# %% [markdown]
# # Spam Email Detection

# %% [markdown]
# ### Import libraries dependencies

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
# load data
raw_mail_data = pd.read_csv('./mail_data.csv')


# %%
# replace null value
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# %%
# label spam mail - 0 and ham mail - 1 
mail_data.loc[mail_data['Category']== 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category']== 'ham', 'Category'] = 1

# %%
# separating data and text
X = mail_data['Message']
Y = mail_data['Category']


# %%
# Split data for train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# %%
# Transform text into feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y train and test value into integer

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# %% [markdown]
# ### Train our model

# %%
model = LogisticRegression()

# train model
model.fit(X_train_features, Y_train)

# prediction
prediction_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_training_data)

# accuracy percent
print(str(round(accuracy_on_training_data * 100, 2)) + '%')

# %%
# Predict on test data
prediction_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_test_data)

# accuracy percent
print(str(round(accuracy_on_test_data * 100, 2)) + '%')

# %% [markdown]
# ### Building a predictive system

# %%
input_mail = ["I HAVE A DATE ON SUNDAY WITH WILL!!"]

# Convert text to feature vector
input_mail_feature = feature_extraction.transform(input_mail)

# Making prediction
prediction = model.predict(input_mail_feature)
if prediction[0] == 1:
    print("It's a ham mail")
else:
    print("It's a spam mail")



