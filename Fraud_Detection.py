import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
def load_data(data_path="./"):
    """Load all necessary datasets"""
    print("Loading datasets...")

    # Load transaction data
    x_train_trans = pd.read_csv("/content/drive/MyDrive/Deep_Brains/train_set/x_train.csv")
    x_val_trans = pd.read_csv("/content/drive/MyDrive/Deep_Brains/validation_set/x_val.csv")
    x_test_trans = pd.read_csv("/content/drive/MyDrive/Deep_Brains/test_set/x_test.csv")

    # Load aggregated data
    x_train_agg = pd.read_csv("/content/drive/MyDrive/Deep_Brains/train_set/x_train_aggregated.csv")
    x_val_agg = pd.read_csv("/content/drive/MyDrive/Deep_Brains/validation_set/x_val_aggregated.csv")
    x_test_agg = pd.read_csv("/content/drive/MyDrive/Deep_Brains/test_set/x_test_aggregated.csv")

    # Load labels
    y_train = pd.read_csv("/content/drive/MyDrive/Deep_Brains/train_set/y_train.csv")
    y_val = pd.read_csv("/content/drive/MyDrive/Deep_Brains/validation_set/y_val.csv")

    # Load student skeleton for submission
    student_skeleton = pd.read_csv("/content/drive/MyDrive/Deep_Brains/student_skeleton.csv")

    print("Datasets loaded successfully!")

    return (x_train_trans, x_train_agg, y_train,
            x_val_trans, x_val_agg, y_val,
            x_test_trans, x_test_agg, student_skeleton)

# Create time-based features
def create_time_features(df):
    """Create time-related features from transaction data"""
    # Create a copy to avoid modifying the original dataframe
    df_time = df.copy()

    # Extract day and hour features
    df_time['Day'] = df_time['Hour'] // 24
    df_time['HourOfDay'] = df_time['Hour'] % 24

    # Define time periods
    df_time['TimePeriod'] = pd.cut(
        df_time['HourOfDay'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )

    # Is weekend (assuming 30 days in a month with day 0 being a Monday)
    df_time['IsWeekend'] = ((df_time['Day'] % 7) >= 5).astype(int)

    return df_time

# Engineer features from transaction data
def engineer_account_features(df):
    """Create account-level features from transaction data"""
    # Group by AccountID
    account_groups = df.groupby('AccountID')

    # Basic count and amount features
    features = pd.DataFrame({
        'TotalTransactions': account_groups.size(),
        'UniqueExternals': account_groups['External'].nunique(),
        'AvgTransactionAmount': account_groups['Amount'].mean(),
        'MedianTransactionAmount': account_groups['Amount'].median(),
        'StdTransactionAmount': account_groups['Amount'].std().fillna(0),
        'MaxAmount': account_groups['Amount'].max(),
        'MinAmount': account_groups['Amount'].min(),
    })

    # Transaction type features
    action_counts = account_groups['Action'].value_counts().unstack().fillna(0)
    if not action_counts.empty:
        action_counts.columns = [f'Count_{col}' for col in action_counts.columns]
        features = features.join(action_counts)

        # Ratio features
        for col in action_counts.columns:
            features[f'Ratio_{col}'] = features[col] / features['TotalTransactions']

        # Add safeguard for division by zero
        cash_out_col = 'Count_CASH_OUT'
        if cash_out_col in features.columns:
            features['CashInToOutRatio'] = features.get('Count_CASH_IN', 0) / features[cash_out_col].replace(0, 0.001)

    # External party type counts
    df['ExternalType'] = df['External'].apply(lambda x: x[0] if isinstance(x, str) and len(x) > 0 else np.nan)
    external_counts = account_groups['ExternalType'].value_counts().unstack().fillna(0)
    if not external_counts.empty:
        external_counts.columns = [f'Count_External_{col}' for col in external_counts.columns]
        features = features.join(external_counts)

        # Ratio of external types
        for col in external_counts.columns:
            features[f'Ratio_{col}'] = features[col] / features['TotalTransactions']

    # Check if UnauthorizedOverdraft column exists in the dataframe
    if 'UnauthorizedOverdraft' in df.columns:
        features['OverdraftRate'] = account_groups['UnauthorizedOverdraft'].mean()
    else:
        print("Warning: 'UnauthorizedOverdraft' column not found in dataframe")
        features['OverdraftRate'] = 0  # Default value

    # Time-based metrics
    df_time = create_time_features(df)
    time_groups = df_time.groupby('AccountID')

    # Transaction timing features
    time_period_counts = time_groups['TimePeriod'].value_counts().unstack().fillna(0)
    if not time_period_counts.empty:
        time_period_counts.columns = [f'Count_{col}' for col in time_period_counts.columns]
        features = features.join(time_period_counts)

        # Ratio of time periods
        for col in time_period_counts.columns:
            features[f'Ratio_{col}'] = features[col] / features['TotalTransactions']

    # Weekend activity
    features['WeekendTransactionRate'] = time_groups['IsWeekend'].mean()

    # Balance volatility features
    features['BalanceVolatility'] = account_groups['NewBalance'].std().fillna(0)

    # Calculate balance velocity (average change in balance)
    balance_velocity = []
    for name, group in account_groups:
        if len(group) > 1:
            # Sort by Hour to ensure chronological order
            sorted_group = group.sort_values('Hour')
            # Calculate absolute differences between consecutive balances
            diffs = np.abs(np.diff(sorted_group['NewBalance'].values))
            velocity = np.mean(diffs) if len(diffs) > 0 else 0
        else:
            velocity = 0
        balance_velocity.append((name, velocity))

    balance_velocity_df = pd.DataFrame(balance_velocity, columns=['AccountID', 'BalanceVelocity']).set_index('AccountID')
    features = features.join(balance_velocity_df)

    # Transaction velocity (transactions per day)
    # Fix the day_counts calculation
    day_transactions = []
    for account_id, group in time_groups:
        # Count transactions per day for each account
        daily_counts = group['Day'].value_counts().reset_index()
        daily_counts.columns = ['Day', 'Transactions']
        daily_counts['AccountID'] = account_id
        day_transactions.append(daily_counts)

    if day_transactions:
        day_counts_df = pd.concat(day_transactions)

        # Calculate metrics per account
        max_daily = day_counts_df.groupby('AccountID')['Transactions'].max()
        avg_daily = day_counts_df.groupby('AccountID')['Transactions'].mean()

        features['MaxDailyTransactions'] = max_daily
        features['AvgDailyTransactions'] = avg_daily
    else:
        features['MaxDailyTransactions'] = 0
        features['AvgDailyTransactions'] = 0

    features['DaysActive'] = time_groups['Day'].nunique()
    features['TransactionDensity'] = features['TotalTransactions'] / features['DaysActive'].replace(0, 1)

    # Round amount features (potential structuring indicator)
    df['IsRoundAmount'] = (df['Amount'] % 100 == 0).astype(int)
    features['RoundAmountRate'] = account_groups['IsRoundAmount'].mean()

    # Last few days activity
    last_week_threshold = df_time['Day'].max() - 7
    last_week_activity = df_time[df_time['Day'] > last_week_threshold].groupby('AccountID').size()
    features['LastWeekTransactions'] = last_week_activity.reindex(features.index, fill_value=0)
    features['LastWeekTransactionRatio'] = features['LastWeekTransactions'] / features['TotalTransactions']

    # Average time between transactions (in hours)
    avg_time_between = []
    for name, group in account_groups:
        if len(group) > 1:
            # Sort by Hour to ensure chronological order
            sorted_hours = sorted(group['Hour'].values)
            # Calculate differences between consecutive timestamps
            diffs = np.diff(sorted_hours)
            avg_diff = np.mean(diffs) if len(diffs) > 0 else 24  # Default to 24 hours if only one transaction
        else:
            avg_diff = 24  # Default to 24 hours if only one transaction
        avg_time_between.append((name, avg_diff))

    avg_time_df = pd.DataFrame(avg_time_between, columns=['AccountID', 'AvgTimeBetweenTransactions']).set_index('AccountID')
    features = features.join(avg_time_df)

    # Transaction patterns
    features['AvgOldBalance'] = account_groups['OldBalance'].mean()
    features['AvgNewBalance'] = account_groups['NewBalance'].mean()

    # Get last balance for each account safely
    last_balances = []
    for name, group in account_groups:
        if not group.empty:
            last_balance = group.sort_values('Hour', ascending=False).iloc[0]['NewBalance']
        else:
            last_balance = 0
        last_balances.append((name, last_balance))

    last_balance_df = pd.DataFrame(last_balances, columns=['AccountID', 'EndingBalance']).set_index('AccountID')
    features = features.join(last_balance_df)

    # Transfer/Payment patterns to catch money laundering patterns
    # Multiple quick transfers to different accounts
    transfer_groups = df[df['Action'] == 'TRANSFER'].groupby('AccountID')
    if len(transfer_groups) > 0:
        unique_recipients = transfer_groups['External'].nunique()
        features['UniqueTransferRecipients'] = unique_recipients

        # Calculate average transfer amount for accounts with transfers
        transfer_amounts = transfer_groups['Amount'].mean()
        features['AvgTransferAmount'] = transfer_amounts

        # Fill missing values (accounts without transfers)
        features['UniqueTransferRecipients'] = features['UniqueTransferRecipients'].fillna(0)
        features['AvgTransferAmount'] = features['AvgTransferAmount'].fillna(0)
    else:
        features['UniqueTransferRecipients'] = 0
        features['AvgTransferAmount'] = 0

    # Clean up NaN values
    features = features.fillna(0)

    return features

# Feature preparation for modeling
def prepare_features(x_trans, x_agg):
    """Prepare features for modeling"""
    # Engineer custom features from transaction data
    print("Engineering features from transaction data...")
    trans_features = engineer_account_features(x_trans)

    # Combine with aggregated features
    print("Combining with pre-aggregated features...")
    x_agg_indexed = x_agg.set_index('AccountID')

    # Check for overlapping columns
    overlapping_columns = set(x_agg_indexed.columns).intersection(set(trans_features.columns))
    if overlapping_columns:
        print(f"Found overlapping columns: {overlapping_columns}")
        # Either rename the columns in one of the DataFrames
        for col in overlapping_columns:
            trans_features = trans_features.rename(columns={col: f"{col}_trans"})

    # Join DataFrames
    combined_features = x_agg_indexed.join(trans_features, how='outer')

    # Handle missing values
    combined_features = combined_features.fillna(0)

    return combined_features

def evaluate_model(model, X, y, model_name, threshold=0.5):
    """Evaluate model performance"""
    # Get predicted probabilities
    y_proba = model.predict_proba(X)[:, 1]

    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Generate metrics
    report = classification_report(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return f1, y_pred, y_proba

def plot_precision_recall(model, X, y, model_name):
    """Plot precision-recall curve"""
    y_scores = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    # Also plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    return pr_auc, roc_auc

def find_optimal_threshold(model, X, y):
    """Find the optimal threshold for classification"""
    # Get predicted probabilities
    y_scores = model.predict_proba(X)[:, 1]

    # Calculate precision and recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y, y_scores)

    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find threshold with maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"F1 score at optimal threshold: {f1_scores[optimal_idx]:.4f}")

    # Plot F1 scores for different thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal threshold: {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_threshold

def scale_features(X_train, X_val=None, X_test=None):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns,
                                  index=X_train.index)

    # Scale validation and test sets if provided
    results = [X_train_scaled]

    if X_val is not None:
        X_val_scaled = pd.DataFrame(scaler.transform(X_val),
                                    columns=X_val.columns,
                                    index=X_val.index)
        results.append(X_val_scaled)

    if X_test is not None:
        X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                     columns=X_test.columns,
                                     index=X_test.index)
        results.append(X_test_scaled)

    return tuple(results)

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train a Random Forest model"""
    print("\nTraining Random Forest...")

    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 15, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    # Create pipeline with SMOTE for handling imbalanced data
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        {f'clf__{key}': val for key, val in param_grid.items()},
        cv=StratifiedKFold(n_splits=3),
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    # Evaluate on validation set
    model = grid_search.best_estimator_
    optimal_threshold = find_optimal_threshold(model, X_val, y_val)
    rf_f1, _, _ = evaluate_model(model, X_val, y_val, "Random Forest", threshold=optimal_threshold)

    # Plot precision-recall curve
    plot_precision_recall(model, X_val, y_val, "Random Forest")

    return model, optimal_threshold

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost model"""
    print("\nTraining XGBoost...")

    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [1, sum(y_train == 0) / (sum(y_train == 1) + 1e-10)]  # For imbalanced data
    }

    # Create XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',  # Add explicit eval_metric to avoid warning
        random_state=42
    )

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        xgb_clf,
        param_grid,
        cv=StratifiedKFold(n_splits=3),
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    # Evaluate on validation set
    model = grid_search.best_estimator_
    optimal_threshold = find_optimal_threshold(model, X_val, y_val)
    xgb_f1, _, _ = evaluate_model(model, X_val, y_val, "XGBoost", threshold=optimal_threshold)

    # Plot precision-recall curve
    plot_precision_recall(model, X_val, y_val, "XGBoost")

    return model, optimal_threshold

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM model"""
    print("\nTraining LightGBM...")

    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50],
        'class_weight': ['balanced']
    }

    # Create LightGBM classifier
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        random_state=42
    )

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        lgb_clf,
        param_grid,
        cv=StratifiedKFold(n_splits=3),
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    # Evaluate on validation set
    model = grid_search.best_estimator_
    optimal_threshold = find_optimal_threshold(model, X_val, y_val)
    lgb_f1, _, _ = evaluate_model(model, X_val, y_val, "LightGBM", threshold=optimal_threshold)

    # Plot precision-recall curve
    plot_precision_recall(model, X_val, y_val, "LightGBM")

    return model, optimal_threshold

def create_ensemble(models, X_val, y_val, thresholds):
    """Create a voting ensemble of models"""
    print("\nCreating ensemble model...")

    # Make predictions with each model
    y_preds = []
    y_probas = []

    for i, (model, threshold) in enumerate(zip(models, thresholds)):
        # Get predicted probabilities
        y_proba = model.predict_proba(X_val)[:, 1]
        y_probas.append(y_proba)

        # Apply optimal threshold
        y_pred = (y_proba >= threshold).astype(int)
        y_preds.append(y_pred)

    # Average the probabilities
    y_proba_ensemble = np.mean(y_probas, axis=0)

    # Find optimal threshold for ensemble
    precision, recall, thresholds_ens = precision_recall_curve(y_val, y_proba_ensemble)

    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find threshold with maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_ens = thresholds_ens[optimal_idx] if optimal_idx < len(thresholds_ens) else 0.5

    # Apply the optimal threshold
    y_pred_ensemble = (y_proba_ensemble >= optimal_threshold_ens).astype(int)

    # Evaluate ensemble
    f1 = f1_score(y_val, y_pred_ensemble)
    print(f"\nEnsemble Performance:")
    print(f"Optimal Threshold: {optimal_threshold_ens:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_val, y_pred_ensemble))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred_ensemble)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return optimal_threshold_ens

def predict_on_test(models, X_test, thresholds, ensemble_threshold):
    """Generate predictions for test data"""
    print("\nGenerating predictions for test data...")

    # Make predictions with each model
    y_probas = []

    for model in models:
        # Get predicted probabilities
        y_proba = model.predict_proba(X_test)[:, 1]
        y_probas.append(y_proba)

    # Average the probabilities
    y_proba_ensemble = np.mean(y_probas, axis=0)

    # Apply the optimal ensemble threshold
    y_pred_ensemble = (y_proba_ensemble >= ensemble_threshold).astype(int)

    return y_pred_ensemble

def main():
    """Main function to train models and generate predictions"""
    # Load all datasets
    x_train_trans, x_train_agg, y_train, x_val_trans, x_val_agg, y_val, x_test_trans, x_test_agg, student_skeleton = load_data()

    # Prepare features for training data
    X_train = prepare_features(x_train_trans, x_train_agg)

    # Prepare features for validation data
    X_val = prepare_features(x_val_trans, x_val_agg)

    # Prepare features for test data
    X_test = prepare_features(x_test_trans, x_test_agg)

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    # Convert target variables to Series with proper index
    y_train_series = y_train.set_index('AccountID')['Fraudster']
    y_val_series = y_val.set_index('AccountID')['Fraudster']

    # Align indices of features and targets
    X_train_aligned = X_train_scaled.loc[y_train_series.index]
    X_val_aligned = X_val_scaled.loc[y_val_series.index]

    # Train models
    rf_model, rf_threshold = train_random_forest(X_train_aligned, y_train_series, X_val_aligned, y_val_series)
    xgb_model, xgb_threshold = train_xgboost(X_train_aligned, y_train_series, X_val_aligned, y_val_series)
    lgb_model, lgb_threshold = train_lightgbm(X_train_aligned, y_train_series, X_val_aligned, y_val_series)

    # Store models and their optimal thresholds
    models = [rf_model, xgb_model, lgb_model]
    thresholds = [rf_threshold, xgb_threshold, lgb_threshold]

    # Create ensemble and evaluate
    ensemble_threshold = create_ensemble(models, X_val_aligned, y_val_series, thresholds)

    # Generate predictions for test data
    test_predictions = predict_on_test(models, X_test_scaled, thresholds, ensemble_threshold)

    # Create submission DataFrame
    account_ids = X_test.index.tolist()
    submission = pd.DataFrame({
        'AccountID': account_ids,
        'Fraudster': test_predictions
    })

    # Ensure all accounts in student_skeleton are included
    skeleton_accounts = student_skeleton['AccountID'].tolist()
    missing_accounts = list(set(skeleton_accounts) - set(account_ids))

    if missing_accounts:
        print(f"Warning: {len(missing_accounts)} accounts in student_skeleton are missing from test predictions.")
        # Use concat instead of append (which is deprecated)
        missing_df = pd.DataFrame({
            'AccountID': missing_accounts,
            'Fraudster': [0] * len(missing_accounts)
        })
        submission = pd.concat([submission, missing_df], ignore_index=True)

    # Sort by AccountID to match student_skeleton order
    submission = submission.set_index('AccountID').reindex(student_skeleton['AccountID']).reset_index()

    # Save submission to file
    submission.to_csv('/content/drive/MyDrive/Deep_Brains/student_submission.csv', index=False)
    print("Submission file created: '/content/drive/MyDrive/Deep_Brains/student_submission.csv'")

    return submission

if __name__ == "__main__":
    main()

# Load the student skeleton file
skeleton_path = "/content/drive/MyDrive/Deep_Brains/student_skeleton.csv"
skeleton_df = pd.read_csv(skeleton_path)

# Display the first few rows of the skeleton file
print("Original skeleton file:")
print(skeleton_df.head())
print(f"Total rows: {len(skeleton_df)}")

# Assuming you've run your model and have predictions
# Option 1: If you have a CSV file with predictions
predictions_path = "/content/drive/MyDrive/Deep_Brains/student_submission.csv"  # Update this path
try:
    predictions_df = pd.read_csv(predictions_path)
    # Create a dictionary mapping AccountID to Fraudster prediction
    predictions_dict = dict(zip(predictions_df['AccountID'], predictions_df['Fraudster']))

    # Update the skeleton dataframe with predictions
    skeleton_df['Fraudster'] = skeleton_df['AccountID'].map(predictions_dict)

except FileNotFoundError:
    # Option 2: If you need to run your model to get predictions
    # This is a placeholder for running your model
    print("Prediction file not found. You'll need to run your model first.")

    # Here you would run the functions from your code to generate predictions
    # For example:
    # from your_module import main
    # result = main()
    # skeleton_df['Fraudster'] = result['Fraudster']

# Make sure all values are either 0 or 1 (no NaNs)
if skeleton_df['Fraudster'].isna().any():
    print(f"Warning: {skeleton_df['Fraudster'].isna().sum()} NaN values found. Filling with 0.")
    skeleton_df['Fraudster'] = skeleton_df['Fraudster'].fillna(0).astype(int)
else:
    skeleton_df['Fraudster'] = skeleton_df['Fraudster'].astype(int)

# Display the updated dataframe
print("\nUpdated skeleton file:")
print(skeleton_df.head())
print(f"Total rows: {len(skeleton_df)}")

# Count of fraud flags
fraud_count = skeleton_df['Fraudster'].sum()
print(f"Number of accounts flagged as fraudulent: {fraud_count} ({fraud_count/len(skeleton_df)*100:.2f}%)")

# Save the updated file
output_path = "/content/drive/MyDrive/Deep_Brains/student_skeleton_submission.csv"
skeleton_df.to_csv(output_path, index=False)
print(f"Updated file saved to: {output_path}")