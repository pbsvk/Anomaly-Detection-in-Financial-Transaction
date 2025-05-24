import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Exploratory Data Analysis

df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

df.head()

df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 'oldbalanceDest': 'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
df.head()

df.isnull().values.any()

df.info()

nan_sum = df['isFraud'].isna().sum()
print(nan_sum)


# ### Summary Statistics
#Summary of Statistics of Numeric Variables
df.describe()

# Summary of Statistics of Categorical Variables
df.describe(include=['object'])

Total_transactions = len(df)
normal = len(df[df.isFraud == 0])
fraudulent = len(df[df.isFraud == 1])
fraud_percentage = round(fraudulent/Total_transactions*100, 2)
normal_percentage = round(normal/Total_transactions*100, 2)
print('Total number of Transactions are {}'.format(Total_transactions))
print('Number of Normal Transactions are {}'.format(normal))
print('Number of fraudulent transactions are {}'.format(fraudulent))
print('Percentage of Normal Transactions is {}'.format(normal_percentage))
print('Percentage of Fraud Transactions is {}'.format(fraud_percentage))


# ### Data Visualization

# Visualize
labels = ["Normal", "Fraud"]
count_classes = df.value_counts(df['isFraud'], sort=True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of fraud and normal transactions")
plt.ylabel("Count")
plt.xlabel("Normal Vs Fraud")
plt.xticks(range(2), labels)
plt.show()

print(df.type.value_counts())

# Visualize the above data
f, ax = plt.subplots(1, 1, figsize=(8,8))
df.type.value_counts().plot(kind='bar', title="Transaction Type", ax=ax, figsize=(8,8))
plt.show()

ax = df.groupby(['type', 'isFraud']).size().plot(kind='bar', figsize=(8,8))
ax.set_title("No. of transactions which are the actual fraud per transaction type")
ax.set_xlabel("(Type, isFraud)")
ax.set_ylabel("Cound of Transaction")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))

fraud_df = df[(df["isFraud"] == 1)]
fraud_df.info()

non_fraud = len(fraud_df[fraud_df.isFraud == 0])
fraud = len(fraud_df[fraud_df.isFraud == 1])
print(non_fraud)
print(fraud)

fraud_df.describe(include=['object'])

new_df = df.loc[df['type'].isin(['CASH_OUT', 'TRANSFER']),:]
print('The new dataframe now has', len(new_df), 'transactions.')

new_df.describe(include=['object'])

trans_0 = new_df[new_df['amount'] == 0]
trans_0


# ### Data Pre-processing

# Remove 0 amount values
new_df = new_df.loc[new_df['amount'] > 0,:]

new_df.info()

new_df_count = len(new_df)
orig_initial_balance = len(new_df[new_df.oldBalanceOrig == 0])
print("Percentage of transactions where originator's initial balance is 0: " + str(round((orig_initial_balance/new_df_count)*100, 2)))
dest_final_balance = len(new_df[new_df.newBalanceDest == 0])
print("Percentage of transactions where destination's final balance is 0: " + str(round(dest_final_balance/new_df_count*100, 2)))

new_df['dest_final_balance'] = new_df['oldBalanceDest'] + new_df['amount']
new_df.head()
new_df['orig_final_balance'] = new_df['oldBalanceOrig'] - new_df['amount']
new_df.head()

new_df['dest_final_balance'] = new_df['oldBalanceDest'] + new_df['amount']
new_df.head()
new_df['orig_final_balance'] = new_df['oldBalanceOrig'] - new_df['amount']
new_df.head()

c1 = len(new_df[new_df.newBalanceDest != new_df.dest_final_balance])
print("Transation where destination balance are not accurately captured: "+ str(round(c1/new_df_count*100, 2)))

c2 = len(new_df[new_df.newBalanceOrig != new_df.orig_final_balance])
print("Transactions where originator balances are not accurately captured: " + str(round(c2/new_df_count*100, 2)))

fraud_trans = len(new_df[new_df.isFraud == 1])
c3 = len(new_df[(new_df.oldBalanceOrig == 0) & (new_df.isFraud == 1)])
print("% of fraudulent transactions where initial balance of orginator is 0: " + str(round(c3/fraud_trans*100, 2)))

gen_trans = len(new_df[new_df.isFraud == 0])
c4 = len(new_df[(new_df.oldBalanceOrig == 0) & (new_df.isFraud == 0)])
print("% of genuine transactions where initial balance of originator is 0: " + str(round(c4/gen_trans*100, 2)))

new_df['origBalance_inacc'] = (new_df['oldBalanceOrig'] - new_df['amount']) - new_df['newBalanceOrig']
new_df['destBalance_inacc'] = (new_df['oldBalanceDest'] + new_df['amount']) - new_df['newBalanceDest']

new_df.head()

new_df = new_df.drop(['nameOrig', 'nameDest','dest_final_balance', 'orig_final_balance', 'isFlaggedFraud'], axis=1)
new_df.info()


# ### Correlation Analysis
import matplotlib.cm as cm

def correlation_plot(df):
    fig = plt.figure(figsize=(10, 10))
    ax1= fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax= ax1.imshow(df.corr(), interpolation = "nearest", cmap = cmap)
    ax1.grid(True)
    plt.title("Correlation Heatmap")
    labels = df.columns.tolist()
    ax1.set_xticklabels(labels, fontsize=13, rotation=45)
    ax1.set_yticklabels(labels, fontsize=13)
    fig.colorbar(cax)
    plt.show()

correlation_plot(new_df)

from scipy.stats import skew, boxcox
from sklearn import preprocessing

new_df['amount_boxcox'] = preprocessing.scale(boxcox(new_df['amount']+1)[0])

figure = plt.figure(figsize=(16, 5))
figure.add_subplot(131)
plt.hist(new_df['amount'], facecolor='blue', alpha=0.75)
plt.xlabel("Transaction amount")
plt.title("Transaction amount")
plt.text(10, 100000, "Skewness: {0:.2f}".format(skew(new_df['amount'])))

figure.add_subplot(132)
plt.hist(np.sqrt(new_df['amount']), facecolor = 'red', alpha=0.5)
plt.xlabel("Square root of amount")
plt.title("Using SQRT on amount")
plt.text(10,100000, "Skewness: {0:.2f}".format(skew(np.sqrt(new_df['amount']))))

figure.add_subplot(133)
plt.hist(new_df['amount_boxcox'], facecolor = "red", alpha=0.5)
plt.xlabel("Boxcox of amount")
plt.title("Using Boxcox on amount")
plt.text(10, 100000, "Skewnes: {0:.2f}".format(skew(new_df['amount_boxcox'])))

plt.show()

new_df['oldBalanceOrig_boxcox'] = preprocessing.scale(boxcox(new_df['oldBalanceOrig']+1)[0])

figure = plt.figure(figsize=(16, 5))
figure.add_subplot(131)
plt.hist(new_df['oldBalanceOrig'], facecolor='blue', alpha=0.75)
plt.xlabel('Old balance originated')
plt.title("Old balance Originated")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(new_df["oldBalanceOrig"])))

figure.add_subplot(132)
plt.hist(np.sqrt(new_df['oldBalanceOrig']), facecolor="red", alpha=0.5)
plt.xlabel("Square root of Old Balance")
plt.title("Square root of old balance originated")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(new_df['oldBalanceOrig']))))

figure.add_subplot(133)
plt.hist(new_df['oldBalanceOrig_boxcox'], facecolor = "red", alpha=0.5)
plt.xlabel("Boxcox for old balance originated")
plt.title("Boxcox on Old Balance Originated")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(new_df['oldBalanceOrig_boxcox'])))

plt.show()

new_df['newbalanceOrg_boxcox'] = preprocessing.scale(boxcox(new_df['newBalanceOrig']+1)[0])

figure = plt.figure(figsize=(16, 5))
figure.add_subplot(131) 
plt.hist(new_df['newBalanceOrig'] ,facecolor='blue',alpha=0.75) 
plt.xlabel("New balance originated") 
plt.title("New balance orgiginated") 
plt.text(2,100000,"Skewness: {0:.2f}".format(skew(new_df['newBalanceOrig'])))


figure.add_subplot(132)
plt.hist(np.sqrt(new_df['newBalanceOrig']), facecolor = 'red', alpha=0.5)
plt.xlabel("Square root of newBal")
plt.title("SQRT on newbalanceOrig")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(new_df['newBalanceOrig']))))

figure.add_subplot(133)
plt.hist(new_df['newbalanceOrg_boxcox'], facecolor = 'red', alpha=0.5)
plt.xlabel("Box cox of newBal")
plt.title("Box cox on newbalanceOrig")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(new_df['newbalanceOrg_boxcox'])))

new_df['oldBalanceDest_boxcox'] = preprocessing.scale(boxcox(new_df['oldBalanceDest']+1)[0])

figure = plt.figure(figsize=(16, 5))
figure.add_subplot(131) 
plt.hist(new_df['oldBalanceDest'] ,facecolor='blue',alpha=0.75) 
plt.xlabel("Old balance Dest") 
plt.title("Old balance Dest") 
plt.text(2,100000,"Skewness: {0:.2f}".format(skew(new_df['oldBalanceDest'])))


figure.add_subplot(132)
plt.hist(np.sqrt(new_df['oldBalanceDest']), facecolor = 'red', alpha=0.5)
plt.xlabel("Square root of oldBal")
plt.title("SQRT on oldbalanceDest")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(new_df['oldBalanceDest']))))

figure.add_subplot(133)
plt.hist(new_df['oldBalanceDest_boxcox'], facecolor = 'red', alpha=0.5)
plt.xlabel("Box cox of oldBal")
plt.title("Box cox on oldbalanceOrig")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(new_df['oldBalanceDest_boxcox'])))

plt.show()

new_df['newBalanceDest_boxcox'] = preprocessing.scale(boxcox(new_df['newBalanceDest']+1)[0])

figure = plt.figure(figsize=(16, 5))
figure.add_subplot(131) 
plt.hist(new_df['newBalanceDest'] ,facecolor='blue',alpha=0.75) 
plt.xlabel("New balance Dest") 
plt.title("New balance Dest") 
plt.text(2,100000,"Skewness: {0:.2f}".format(skew(new_df['newBalanceDest'])))


figure.add_subplot(132)
plt.hist(np.sqrt(new_df['newBalanceDest']), facecolor = 'red', alpha=0.5)
plt.xlabel("Square root of newBal")
plt.title("SQRT on newbalanceDest")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(np.sqrt(new_df['newBalanceDest']))))

figure.add_subplot(133)
plt.hist(new_df['newBalanceDest_boxcox'], facecolor = 'red', alpha=0.5)
plt.xlabel("Box cox of newBal")
plt.title("Box cox on newbalanceDest")
plt.text(2, 100000, "Skewness: {0:.2f}".format(skew(new_df['newBalanceDest_boxcox'])))

plt.show()

new_df.head()

print("The fraud transaction of the filtered dataset: {0:.4f}%".format((len(new_df[new_df.isFraud == 1])/len(new_df))*100))

new_df.drop(["oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest", "amount", "type", "origBalance_inacc", "destBalance_inacc", "step"], axis=1, inplace=True)
new_df.head()

new_df.info()

X = new_df.iloc[:, new_df.columns != 'isFraud']
y = new_df.iloc[:, new_df.columns == 'isFraud']

#Number of data points in the minoroity class
number_records_fraud = len(new_df[new_df.isFraud == 1])
fraud_indices = new_df[new_df.isFraud == 1].index.values

# Picking the indices of the normal classes
normal_indices = new_df[new_df.isFraud == 0].index

# Out of the indices we picked, randomly select "x" number (x - same as total fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = new_df.loc[under_sample_indices, :]

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'isFraud']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'isFraud']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.isFraud == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.isFraud == 1])/len(under_sample_data))
print("Total number of transactions in resample data: ", len(under_sample_data))


# ### Modeling
from sklearn.model_selection import train_test_split

## Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

print("Number of transactions in train dataset: ", format(len(X_train), ",d"))
print("Number of transactions in test dataset: ", format(len(X_test), ",d"))
print("Total number of transactions: ", format(len(X_train)+len(X_test),",d"))

# Undersampled Dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size = 0.3, random_state=0)

print("Number transactions train dataset: ", format(len(X_train_undersample),',d'))
print("Number transactions test dataset: ", format(len(X_test_undersample),',d'))
print("Total number of transactions: ", format(len(X_train_undersample)+len(X_test_undersample),',d'))


# ### Logistic Regression
from sklearn.linear_model import LogisticRegression
from  sklearn import metrics
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_undersample = label_encoder.fit_transform(y_train_undersample.values.ravel())
y_test_undersample = label_encoder.fit_transform(y_test_undersample.values.ravel())

logreg = LogisticRegression()
logreg.fit(X_train_undersample, y_train_undersample)

y_pred = logreg.predict(X_test_undersample)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_undersample, y_test_undersample)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test_undersample, y_pred)
print(confusion_matrix)

import seaborn as sns

ax = sns.heatmap(confusion_matrix,fmt=".2f", annot=True, cmap="Blues")

ax.set_title('Seaborn Confusion Matrix with Labels\n\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')

ax.xaxis.set_ticklabels([1,0])
ax.yaxis.set_ticklabels([1,0])

plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test_undersample, y_pred))

from sklearn.metrics import roc_auc_score, roc_curve

logit_roc_auc = roc_auc_score(y_test_undersample, logreg.predict(X_test_undersample))
fpr, tpr, thresholds = roc_curve(y_test_undersample, logreg.predict_proba(X_test_undersample)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# ### Feed Forward Neural Network
get_ipython().system('pip install tensorflow')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the neural network architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_undersample.shape[1],)),
    Dropout(0.5),  # Add dropout for regularization
    Dense(32, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_undersample, y_train_undersample, epochs=20, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_undersample, y_test_undersample)
print('Accuracy of the neural network on test set: {:.2f}'.format(accuracy))

# Predictions
y_pred_proba = model.predict(X_test_undersample)
y_pred = (y_pred_proba > 0.5).astype(int)

# Confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test_undersample, y_pred)
print('Confusion Matrix:\n', conf_matrix)

# Plotting confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report
from sklearn.metrics import classification_report
print('Classification Report:\n', classification_report(y_test_undersample, y_pred))

# ROC curve
from sklearn.metrics import roc_auc_score, roc_curve

roc_auc = roc_auc_score(y_test_undersample, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test_undersample, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# ### Estimation of parameters

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the PaySim dataset
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Select only numerical columns for estimation
numerical_columns = df.select_dtypes(include=['number']).columns

# Loop through each numerical column (except isFraud and isFlaggedFraud)
for col in numerical_columns:
    if col not in ['isFraud', 'isFlaggedFraud']:
        # Define the independent variable (X) and dependent variable (y)
        X = df[[col]]
        y = df['isFraud']

        # Add a constant term to the independent variable (for intercept estimation)
        X = sm.add_constant(X)

        # Fit the linear regression model
        model = sm.OLS(y, X).fit()

        # Calculate R-squared (R2)
        R2 = model.rsquared

        # Calculate log-likelihood (MLE)
        MLE = model.llf

        # Predict the target variable
        y_pred = model.predict(X)

        # Calculate root mean squared error (RMSE)
        RMSE = np.sqrt(mean_squared_error(y, y_pred))

        # Calculate mean squared error (MSE)
        MSE = mean_squared_error(y, y_pred)

        # Print the evaluation metrics
        print(f"Evaluation metrics for {col}:")
        print(f"R-squared (R2): {R2}")
        print(f"Log-likelihood (MLE): {MLE}")
        print(f"Root Mean Squared Error (RMSE): {RMSE}")
        print(f"Mean Squared Error (MSE): {MSE}")
        print()
