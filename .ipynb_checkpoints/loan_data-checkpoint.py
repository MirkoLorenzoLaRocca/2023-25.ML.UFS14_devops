# Importing libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def pippo(path):
    output_dir = os.environ['SM_MODEL_DIR']
    df = pd.read_csv(path)
    df.head()
    
    # Plot each selected column as a boxplot against the target variable
    selected_columns = ['loan_percent_income']
    target = 'loan_status'

    plt.figure(figsize=(12, 8))
    for i, col in enumerate(selected_columns, 1):
        plt.subplot(1, len(selected_columns), i)
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f'Box Plot of {col} by Target {target}')
    plt.tight_layout()
    plt.savefig(output_dir + "/box_plot.png")  # Save instead of showing
    plt.close()

    # Prepare the data
    df_binary = df.copy()
    df_binary['person_gender'] = df['person_gender'].astype(str).str.strip().str.lower().map({'female': 0, 'male': 1})
    df_binary['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    
    # One hot encoding
    df_encoded = pd.get_dummies(df_binary, columns=['person_education'], prefix='edu', dtype=int)
    df_encoded['edu_High_School'] = df_encoded['edu_High School']
    df_encoded = df_encoded.drop(columns=['edu_High School'])
    df_encoded = pd.get_dummies(df_encoded, columns=['person_home_ownership'], prefix="home_own", dtype=int)
    df_encoded = pd.get_dummies(df_encoded, columns=['loan_intent'], dtype=int)

    # Scaling
    columns_to_scale = [
        'person_age', 'person_income', 'loan_amnt', 'credit_score', 'loan_int_rate'
    ]
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])

    # Train-test split
    from sklearn.model_selection import train_test_split
    y = df_scaled['loan_status']
    x = df_scaled.drop(['loan_status'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)
    
    # Logistic Regression Weights
    weights = pd.DataFrame()
    weights['Features'] = x.columns
    weights['Weights'] = lr.coef_.reshape(-1)
    weights.sort_values(by='Weights', ascending=False).to_csv("logistic_regression_weights.csv", index=False)

    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # XGBoost Classifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    
    # XGBoost Feature Importance Plot
    sorted_idx = xgb.feature_importances_.argsort()
    plt.figure()
    plt.barh(x.columns[sorted_idx], xgb.feature_importances_[sorted_idx])
    plt.xlabel('XGBoost feature importance')
    plt.savefig(output_dir + "/xgboost_feature_importance.png")  # Save instead of showing
    plt.close()

    # Contingency Table Plot
    contingency_table = pd.crosstab(df['previous_loan_defaults_on_file'], df['loan_status'])
    contingency_table.plot(kind='bar', stacked=True, color=['#FF9999', '#66B3FF'])
    plt.title('Loan Approval Status by Previous Loans')
    plt.xlabel('Loans in past')
    plt.ylabel('Approved loans count')
    plt.xticks(ticks=[0, 1], labels=['No loans in past', 'Had failed loans'], rotation=0)
    plt.legend(title='Approved loan', labels=['Not approved', 'Approved'])
    plt.tight_layout()
    plt.savefig(output_dir + "/loan_approval_status.png")  # Save instead of showing
    plt.close()
