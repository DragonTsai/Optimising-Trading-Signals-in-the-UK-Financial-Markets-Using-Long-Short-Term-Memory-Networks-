#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import yfinance as yf
import pandas as pd
#iSharesCore_FTSE100 = yf.Ticker("ISF.L")
#data = iSharesCore_FTSE100.history(start="2010-01-01", end="2024-04-15")
#data.to_csv("iSharesCore_FTSE100.csv")


# In[ ]:


# Read the CSV file, which contains data related to the iShares Core FTSE 100 ETF.
ISFframe = pd.read_csv("D:/桌面/iSharesCore_FTSE100.csv")
ISFframe.head()


# In[ ]:


# Importing the pandas_ta library for technical analysis functions
import pandas_ta as ta  
# Calculate Relative Strength Index (RSI) and append it to ISFframe
ISFframe.ta.rsi(close='Close', length=14, append=True)
# Calculate Moving Average Convergence Divergence (MACD) and append it to ISFframe
ISFframe.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
# Calculate Bollinger Bands (BBANDS) and append it to ISFframe
ISFframe.ta.bbands(close='Close', length=20, std=2, append=True)
ISFframe.head()


# In[ ]:


# Remove rows with missing values from ISFframe and assign the result to ISFframe_new
ISFframe_new = ISFframe.dropna(axis=0)
# Remove 'Stock Splits' and 'Capital Gains' columns from ISFframe_new because they are not useful
del ISFframe_new['Stock Splits']
del ISFframe_new['Capital Gains']
# Convert the 'Date' column to datetime and extract only the date part
ISFframe_new['Date'] = pd.to_datetime(ISFframe_new['Date'], utc=True).dt.date
#ISFframe_new.set_index('Date', inplace = True)
ISFframe_new.head()


# In[ ]:


# Calculate the short-term moving average (short_ma) using a window of 10 periods on the 'Close' column
ISFframe_new["short_ma"] = ISFframe_new["Close"].rolling(window=10, min_periods=1, center=False).mean()
# Calculate the long-term moving average (long_ma) using a window of 20 periods on the 'Close' column
ISFframe_new["long_ma"] = ISFframe_new["Close"].rolling(window=20, min_periods=1, center=False).mean()
# Calculate the short-term momentum (short_mom) using a window of 10 periods on the 'RSI_14' column
ISFframe_new["short_mom"] = ISFframe_new["RSI_14"].rolling(window=10, min_periods=1, center=False).mean()
# Calculate the long-term momentum (long_mom) using a window of 20 periods on the 'RSI_14' column
ISFframe_new["long_mom"] = ISFframe_new["RSI_14"].rolling(window=20, min_periods=1, center=False).mean()
ISFframe_new.head()


# In[ ]:


import numpy as np  
# Create a new column 'label' in ISFframe_new where 1 indicates short_ma > long_ma, otherwise 0
ISFframe_new['label'] = np.where(ISFframe_new["short_ma"] > ISFframe_new["long_ma"], 1, 0)
ISFframe_new.head()


# In[ ]:


import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(8, 4))
# Count the frequency of each label in the 'label' column
label_counts = ISFframe_new['label'].value_counts()
# Plot a bar chart of label frequencies 
plot = label_counts.plot(kind='bar', color='deepskyblue', title='Frequency of Labels')
plt.title('Data Distribution')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


# Read the UK unemployment rate data downloaded from Office for National Statistics 
uk_unemploymentrate = pd.read_csv("D:/桌面/UK unemployment rate.csv")
uk_unemploymentrate.tail()


# In[ ]:


def month_to_quarter(month):
    # Dictionary mapping months to quarters
    month_to_q = {
        'JAN': 'Q1', 'FEB': 'Q1', 'MAR': 'Q1',
        'APR': 'Q2', 'MAY': 'Q2', 'JUN': 'Q2',
        'JUL': 'Q3', 'AUG': 'Q3', 'SEP': 'Q3',
        'OCT': 'Q4', 'NOV': 'Q4', 'DEC': 'Q4'
    }
    # Return the corresponding quarter for the given month abbreviation
    return month_to_q.get(month, None)

# Apply the month_to_quarter function to the 'Unit' column to derive the quarter
uk_unemploymentrate['Quarter'] = uk_unemploymentrate['Unit'].apply(
    lambda x: f"{x.split()[0]} {month_to_quarter(x.split()[1])}" if len(x.split()) > 1 else x
)

# Filter out rows where Quarter is NaN or contains 'None'
uk_unemploymentrate = uk_unemploymentrate[uk_unemploymentrate['Quarter'].notna() & ~uk_unemploymentrate['Quarter'].str.contains('None')]
# Group the data by Quarter and calculate the mean for numeric columns
uk_unemploymentrate = uk_unemploymentrate.groupby('Quarter').mean(numeric_only=True).reset_index()
# Filter the data for the range '2010 Q1' to '2023 Q1' inclusive
filtered_data1 = uk_unemploymentrate[(uk_unemploymentrate['Quarter'] >= '2010 Q1') & (uk_unemploymentrate['Quarter'] <= '2023 Q1')]
filtered_data1.rename(columns={'%': 'Unemployment Rate in the UK(%)'}, inplace=True)
filtered_data1.head()


# In[ ]:


# Read the UK consumer price inflation data from Office for National Statistics 
uk_consumerpriceinflation = pd.read_csv("D:/桌面/UK consumer price inflation.csv")
uk_consumerpriceinflation.rename(columns={'Unit': 'Quarter', '%': 'Consumer Price Inflation in the UK(%)'}, inplace=True)
# Filter the data for the range '2010 Q1' to '2023 Q1' inclusive
filtered_data2 = uk_consumerpriceinflation[(uk_consumerpriceinflation['Quarter'] >= '2010 Q1') & (uk_consumerpriceinflation['Quarter'] <= '2023 Q1')]
filtered_data2.head()


# In[ ]:


# Read the UK GDP data from Office for National Statistics
uk_gdp = pd.read_csv("D:/桌面/UK GDP.csv")
uk_gdp.rename(columns={'Unit': 'Quarter', '%': 'GDP in the UK(%)'}, inplace=True)
# Filter the data for the range '2010 Q1' to '2023 Q1' inclusive
filtered_data3 = uk_gdp[(uk_gdp['Quarter'] >= '2010 Q1') & (uk_gdp['Quarter'] <= '2023 Q1')]
filtered_data3.head()


# In[ ]:


# Merge filtered_data1 and filtered_data2 on the 'Quarter' column using an outer join
merged_data = pd.merge(filtered_data1, filtered_data2, on='Quarter', how='outer')
# Merge merged_data with filtered_data3 on the 'Quarter' column using an outer join
macrofactors_data = pd.merge(merged_data, filtered_data3, on='Quarter', how='outer')
macrofactors_data.head()


# In[ ]:


# Convert the 'Date' column in ISFframe_new to datetime format and set it as the index
ISFframe_new['Date'] = pd.to_datetime(ISFframe_new['Date'])
ISFframe_new.set_index('Date', inplace=True)
# Convert the 'Quarter' column in macrofactors_data to datetime format and set it as the index
macrofactors_data['Quarter'] = pd.to_datetime(macrofactors_data['Quarter'].str.replace(' Q', '-Q'), errors='coerce')
macrofactors_data.set_index('Quarter', inplace=True)
# Resample macrofactors_data to daily frequency, forward fill missing values, and reindex based on ISFframe_new's index
daily_macro_data = macrofactors_data.resample('D').ffill().reindex(ISFframe_new.index, method='ffill')
# Combine ISFframe_new and daily_macro_data based on their indexes
combined_data = ISFframe_new.join(daily_macro_data)
combined_data.tail(50)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Drop the 'label' column to extract features
features = combined_data.drop('label', axis=1)
# Extract labels into a DataFrame
labels = pd.DataFrame({"label": combined_data.label})

# Split the data into training and testing sets (70% training, 30% testing)
split = int(len(combined_data) * 0.7)
svm_X_train = features.iloc[:split, :].copy()
svm_X_test = features.iloc[split:, :].copy()
svm_y_train = labels.iloc[:split, :].copy()
svm_y_test = labels.iloc[split:, :].copy()

# Further split the training data to include a validation set (20% of training data)
svm_X_train, X_val, svm_y_train, y_val = train_test_split(svm_X_train, svm_y_train, test_size=0.20, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(svm_X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(svm_X_test)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, svm_y_train['label'].values.ravel())  # Corrected to convert to NumPy array then ravel

# Evaluate the model on the validation set
y_val_pred = svm_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred)

# Print validation accuracy and classification report
print(val_accuracy)
print(val_report)


# In[ ]:


# Predict the labels for the test set
svm_predictions = svm_model.predict(svm_X_test)
# Calculate the accuracy
accuracy = accuracy_score(svm_y_test, svm_predictions)
print("SVM model accuracy for test dataset:", accuracy)


# In[ ]:


# Drop the 'label' column to extract features
# Extract labels into a DataFrame
X = combined_data.drop('label', axis=1)
from sklearn.preprocessing import StandardScaler
X[X.columns] = StandardScaler().fit_transform(X[X.columns])
y = pd.DataFrame({"label": combined_data.label})

# Split the data into training and testing sets (70% training, 30% testing)
split = int(len(combined_data) * 0.7)
train_X = X.iloc[:split, :].copy()
test_X = X.iloc[split:].copy()
train_y = y.iloc[:split, :].copy()
test_y = y.iloc[split:].copy()

# Reshape the data for LSTM input
X_train, y_train, X_test, y_test = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# No reshaping needed for y_train and y_test if the last layer is a Dense layer expecting a single output
# Since y_train is already an array of shape (n_samples, 1), we don't need to reshape it further

# Import required libraries for building the LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization

# Initialize and build the model
regressor = Sequential()
regressor.add(LSTM(units=35, return_sequences=True, input_shape=(1, X_train.shape[2])))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=35, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=35, return_sequences=False))  # Note the change here to not return sequences
regressor.add(Dropout(0.3))
regressor.add(Dense(units=1, activation="sigmoid"))  # Output layer for binary classification

# Compile the model
regressor.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

# Display the model summary
regressor.summary()


# In[ ]:


# Train the LSTM model on the training data
train_history = regressor.fit(X_train, y_train,
                               batch_size=200,  # Number of samples per gradient update
                               epochs=100,  # Number of epochs to train the model
                               verbose=2,  # Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)


# In[ ]:


# Extract loss values
loss = train_history.history["loss"]
val_loss = train_history.history["val_loss"]
plt.figure(figsize=(10, 6))  

# Plot training loss
plt.plot(loss, label="Training Loss", color="blue", linestyle="-", linewidth=2)
# Plot validation loss
plt.plot(val_loss, label="Validation Loss", color="red", linestyle="--", linewidth=2)

plt.title("Model Loss", fontsize=16)  
plt.xlabel("Epoch", fontsize=14)  
plt.ylabel("Loss", fontsize=14)  
plt.legend(fontsize=12)  
plt.grid(True)  # Add grid lines for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()  


# In[ ]:


from tqdm.notebook import tqdm
results = []
print('Computing LSTM feature importance...')


if X_test.ndim == 2:  
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

oof_preds = regressor.predict(X_test, verbose=0).squeeze()
baseline_mae = np.mean(np.abs(oof_preds - y_test.squeeze()))
results.append({'feature': 'BASELINE', 'mae': baseline_mae})

# Iterate over each feature to shuffle
for k in tqdm(range(X_test.shape[2])):  
    
    save_col = X_test[:, :, k].copy()
    np.random.shuffle(X_test[:, :, k])
    oof_preds = regressor.predict(X_test, verbose=0).squeeze()
    mae = np.mean(np.abs(oof_preds - y_test.squeeze()))
    results.append({'feature': test_X.columns[k], 'mae': mae})
    
    # Restore original data
    X_test[:, :, k] = save_col


# In[ ]:


from tqdm.notebook import tqdm  # Importing tqdm for progress visualization
results = []  
print('Computing LSTM feature importance...') 

# Reshape X_test if its dimension is 2
if X_test.ndim == 2:
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Predict using the trained regressor and calculate baseline MAE
oof_preds = regressor.predict(X_test, verbose=0).squeeze()
baseline_mae = np.mean(np.abs(oof_preds - y_test.squeeze()))
results.append({'feature': 'BASELINE', 'mae': baseline_mae})  # Store baseline MAE

# Iterate over each feature to shuffle and compute MAE
for k in tqdm(range(X_test.shape[2])):  # Loop over each feature
    save_col = X_test[:, :, k].copy()  # Save the original data of the feature
    np.random.shuffle(X_test[:, :, k])  # Shuffle the feature
    oof_preds = regressor.predict(X_test, verbose=0).squeeze()  # Predict using shuffled data
    mae = np.mean(np.abs(oof_preds - y_test.squeeze()))  # Calculate MAE
    results.append({'feature': test_X.columns[k], 'mae': mae})  # Store MAE for the feature
    X_test[:, :, k] = save_col  # Restore original data of the feature


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(results)
df = df.sort_values('mae')
plt.figure(figsize=(8,10))

# Set the color of the bars to green
plt.barh(np.arange(len(list(test_X.columns))+1), df.mae, color='green')

plt.yticks(np.arange(len(list(test_X.columns))+1), df.feature.values)
plt.title('LSTM Feature Importance', size=16)
plt.ylim((-1, len(list(test_X.columns))+1))
plt.plot([baseline_mae, baseline_mae], [-1, len(list(test_X.columns))+1], '--', color='red',
         label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
plt.xlabel(f'MAE with feature permuted', size=14)
plt.ylabel('Feature', size=14)
plt.legend()
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a DataFrame from the results computed earlier 
df = pd.DataFrame(results)
# Sort the DataFrame by 'mae' column
df = df.sort_values('mae')
plt.figure(figsize=(8, 10))
plt.barh(np.arange(len(list(test_X.columns)) + 1), df.mae, color='green')
plt.yticks(np.arange(len(list(test_X.columns)) + 1), df.feature.values)
plt.title('LSTM Feature Importance', size=16)
plt.ylim((-1, len(list(test_X.columns)) + 1))
plt.plot([baseline_mae, baseline_mae], [-1, len(list(test_X.columns)) + 1], '--', color='red',
         label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
plt.xlabel('MAE with feature permuted', size=14)
plt.ylabel('Feature', size=14)
plt.legend()
plt.show()


# In[ ]:


# Use the SVM model to make predictions on the scaled test data
svm_predict = svm_model.predict(X_test_scaled)
# Create a DataFrame to store the results
svm_result = pd.DataFrame({"Close": combined_data.iloc[split:]["Close"]})
# Add the real labels from svm_y_test to the DataFrame
svm_result["Real"] = svm_y_test['label']
# Add the SVM predictions to the DataFrame
svm_result["SVM Predict"] = list(svm_predict)
svm_result.tail(50)


# In[ ]:


predict_x = regressor.predict(X_test) 
df_predict = pd.DataFrame(predict_x,columns = ["Buy"])
df_predict["Action"] = np.where(df_predict["Buy"] > 0.5, 1, 0)
result = pd.DataFrame({"Close":combined_data.iloc[split:]["Close"]})
result["Real"] = test_y["label"]
result["LSTM Predict"] = list(df_predict["Action"])
result.tail(50)


# In[ ]:


# Use the LSTM model to make predictions on the test data
predict_x = regressor.predict(X_test)
# Create a DataFrame to store the LSTM predictions
df_predict = pd.DataFrame(predict_x, columns=["Buy"])
# Convert the predictions to binary actions based on a threshold of 0.5
df_predict["Action"] = np.where(df_predict["Buy"] > 0.5, 1, 0)
# Create a DataFrame to store the results
result = pd.DataFrame({"Close": combined_data.iloc[split:]["Close"]})
# Add the real labels from test_y to the DataFrame
result["Real"] = test_y["label"]
# Add the LSTM predictions to the DataFrame
result["LSTM Predict"] = list(df_predict["Action"])
result.tail(50)


# In[ ]:


print("SVM model accuracy for test dataset:", accuracy)  # Print SVM model accuracy for the test dataset
print("LSTM model accuracy for test dataset ")  # Print LSTM model accuracy for the test dataset
# Evaluate the LSTM model on the test dataset and print the results
evaluation_results = regressor.evaluate(X_test, y_test, verbose=1)


# In[ ]:


backtest = result.copy()

backtest['daily_returns'] = backtest['Close'].pct_change()  

backtest['LSTM predict_strategy_returns'] = backtest['daily_returns'] * backtest['LSTM Predict'].shift(1)
backtest['SVM predict_strategy_returns'] = backtest['daily_returns'] * svm_result['SVM Predict'].shift(1)
backtest['real_strategy_returns'] = backtest['daily_returns'] * backtest['Real'].shift(1)

backtest['cumulative_LSTM predict_returns'] = (1 + backtest['LSTM predict_strategy_returns']).cumprod()
backtest['cumulative_SVM predict_returns'] = (1 + backtest['SVM predict_strategy_returns']).cumprod()
backtest['cumulative_real_returns'] = (1 + backtest['real_strategy_returns']).cumprod()
backtest['cumulative_market_returns'] = (1 + backtest['daily_returns']).cumprod()

plt.figure(figsize=(14, 7))
plt.plot(backtest.index, backtest['cumulative_LSTM predict_returns'], label='LSTM Predict Label Strategy')
plt.plot(backtest.index, backtest['cumulative_SVM predict_returns'], label='SVM Predict Label Strategy')
plt.plot(backtest.index, backtest['cumulative_real_returns'], label='Real Label Strategy')
plt.plot(backtest.index, backtest['cumulative_market_returns'], label='Market Performance')
plt.title('Strategy Backtesting')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()


# In[ ]:


# Create a copy of the result DataFrame for backtesting
backtest = result.copy()

# Calculate daily returns
backtest['daily_returns'] = backtest['Close'].pct_change()

# Calculate strategy returns based on LSTM predictions, SVM predictions, and real labels
backtest['LSTM predict_strategy_returns'] = backtest['daily_returns'] * backtest['LSTM Predict'].shift(1)
backtest['SVM predict_strategy_returns'] = backtest['daily_returns'] * svm_result['SVM Predict'].shift(1)
backtest['real_strategy_returns'] = backtest['daily_returns'] * backtest['Real'].shift(1)

# Calculate cumulative returns for each strategy
backtest['cumulative_LSTM predict_returns'] = (1 + backtest['LSTM predict_strategy_returns']).cumprod()
backtest['cumulative_SVM predict_returns'] = (1 + backtest['SVM predict_strategy_returns']).cumprod()
backtest['cumulative_real_returns'] = (1 + backtest['real_strategy_returns']).cumprod()
backtest['cumulative_market_returns'] = (1 + backtest['daily_returns']).cumprod()

# Plot cumulative returns for each strategy and market performance
plt.figure(figsize=(14, 7))
plt.plot(backtest.index, backtest['cumulative_LSTM predict_returns'], label='LSTM Predict Label Strategy')
plt.plot(backtest.index, backtest['cumulative_SVM predict_returns'], label='SVM Predict Label Strategy')
plt.plot(backtest.index, backtest['cumulative_real_returns'], label='Real Label Strategy')
plt.plot(backtest.index, backtest['cumulative_market_returns'], label='Market Performance')
plt.title('Strategy Backtesting')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()


# In[ ]:


# Extract the last row of the DataFrame containing cumulative returns for each strategy and market performance
backtest[['cumulative_LSTM predict_returns','cumulative_SVM predict_returns','cumulative_real_returns','cumulative_market_returns']][-1:]

