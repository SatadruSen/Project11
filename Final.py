import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Loading training features and labels
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')

# Separating features and labels
X = train_features.drop(columns=['respondent_id'])
y = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

# Finding categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Pre-processing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

# Pre-processing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Model Definition
model_xyz = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(max_iter=1000))])

model_seasonal = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(max_iter=1000))])

# Splitting the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model_xyz.fit(X_train, y_train['xyz_vaccine'])
model_seasonal.fit(X_train, y_train['seasonal_vaccine'])

# Predicting probabilities
y_pred_xyz = model_xyz.predict_proba(X_val)[:, 1]
y_pred_seasonal = model_seasonal.predict_proba(X_val)[:, 1]

# Evaluating the models
roc_auc_xyz = roc_auc_score(y_val['xyz_vaccine'], y_pred_xyz)
roc_auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], y_pred_seasonal)
mean_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2

print(f"ROC AUC for xyz_vaccine: {roc_auc_xyz}")
print(f"ROC AUC for seasonal_vaccine: {roc_auc_seasonal}")
print(f"Mean ROC AUC: {mean_roc_auc}")

# Test Set Load
test_features = pd.read_csv('test_set_features.csv')

# Ensuring the test set has the same preprocessing applied
test_pred_xyz = model_xyz.predict_proba(test_features.drop(columns=['respondent_id']))[:, 1]
test_pred_seasonal = model_seasonal.predict_proba(test_features.drop(columns=['respondent_id']))[:, 1]

# Preparing the submission file
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': test_pred_xyz,
    'seasonal_vaccine': test_pred_seasonal
})

submission.to_csv('submission_format.csv', index=False)
