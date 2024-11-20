from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# FlagMissing transformer 
class FlagMissing(BaseEstimator, TransformerMixin):
    """
    Flags missing values by creating new columns for each feature indicating if a value is missing.
    This helps track where data was missing for future analysis.
    """
    def __init__(self, columns_to_flag):
        self.columns_to_flag = columns_to_flag
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Avoid modifying original data during transformation
        
        # Create a separate DataFrame for the flags
        missing_flags = pd.DataFrame()
        
        for col in self.columns_to_flag:
            missing_flags[f'FLAG_{col}'] = np.where(X[col].isnull(), 0, 1)
        
        # Concatenate the flags to the original DataFrame (this is done to avoid fragmentation)
        X = pd.concat([X, missing_flags], axis=1)
        
        return X

# Custom transformer to impute missing values with specified strategies
class ImputeColumns(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using different strategies:
    - Median for selected columns.
    - 'Unknown' for selected categorical columns.
    - Zero for other selected columns.
    """
    def __init__(self, unknown_columns=None):
        self.unknown_columns = unknown_columns
  
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Impute with 'Unknown' for specified categorical columns
        if self.unknown_columns:
            for col in self.unknown_columns:
                X[col].fillna('Unknown', inplace=True)
        
        return X

# Custom transformer to cap outliers before scaling, ensuring percentiles are only calculated once
class CapOutliers(BaseEstimator, TransformerMixin):
    """
    Caps outliers for numerical columns by setting values above the specified percentiles.
    The percentiles are calculated once during the fit and applied consistently during transforms.
    """
    def __init__(self, columns_to_cap, lower_percentile=2, upper_percentile=98):
        self.columns_to_cap = columns_to_cap
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_caps_ = {}
        self.upper_caps_ = {}
    
    def fit(self, X, y=None):
        # Calculate and store the caps for each column during fit
        for col in self.columns_to_cap:
            self.lower_caps_[col] = X[col].quantile(self.lower_percentile / 100)
            self.upper_caps_[col] = X[col].quantile(self.upper_percentile / 100)
        return self
    
    def transform(self, X):
        X = X.copy()  # Work on a copy to avoid modifying the original dataframe
        # Apply the pre-calculated caps during transform
        for col in self.columns_to_cap:
            X[col] = X[col].clip(lower=self.lower_caps_[col], upper=self.upper_caps_[col])
        return X

# Custom transformer to scale numerical columns
class ScaleNumericFeatures(BaseEstimator, TransformerMixin):
    """
    Scales numerical features using StandardScaler.
    This ensures that all features are on a similar scale, which helps models converge faster.
    """
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X

# Custom transformer to impute remaining missing values with mean for numeric columns
class ImputeWithMedian(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for numeric columns using the mean of each column.
    This is useful for handling the last few missing values in numeric columns.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        return X

# Custom transformer to impute remaining missing values with mode for non-numeric columns
class ImputeWithMode(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for non-numeric columns using the most frequent value (mode).
    This is useful for categorical data.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(exclude=['float64', 'int64']).columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)
        return X



#now specifically bringing together a pipeline for this dataset
#nmar suspects
flag_cols = [
    'COMMONAREA_MEDI', 'COMMONAREA_AVG', 'COMMONAREA_MODE',
    'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MEDI',
    'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_AVG', 'LIVINGAPARTMENTS_MEDI',
    'FLOORSMIN_AVG', 'FLOORSMIN_MODE', 'FLOORSMIN_MEDI',
    'YEARS_BUILD_MEDI', 'YEARS_BUILD_MODE', 'YEARS_BUILD_AVG',
    'LANDAREA_MEDI', 'LANDAREA_MODE', 'LANDAREA_AVG',
    'BASEMENTAREA_MEDI', 'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE',
    'NONLIVINGAREA_MODE', 'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MEDI',
    'ELEVATORS_MEDI', 'ELEVATORS_AVG', 'ELEVATORS_MODE',
    'APARTMENTS_MEDI', 'APARTMENTS_AVG', 'APARTMENTS_MODE',
    'ENTRANCES_MEDI', 'ENTRANCES_AVG', 'ENTRANCES_MODE',
    'LIVINGAREA_AVG', 'LIVINGAREA_MODE', 'LIVINGAREA_MEDI',
    'HOUSETYPE_MODE', 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI', 'FLOORSMAX_AVG',
    'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BEGINEXPLUATATION_AVG',
    'TOTALAREA_MODE',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]

numerical_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

unknown_columns = ['NAME_TYPE_SUITE', 'OCCUPATION_TYPE']

# Create the pipeline that chains all the transformations together
application_cleaning_pipeline = Pipeline(steps=[
    ('flag_missing', FlagMissing(columns_to_flag=flag_cols)),
    ('unknowns', ImputeColumns( unknown_columns=unknown_columns)),
    ('cap_outliers', CapOutliers(columns_to_cap=numerical_cols)),  # Cap outliers
    ('mean_impute', ImputeWithMedian()),   # Impute remaining numeric columns with median
    ('mode_impute', ImputeWithMode())    # Impute remaining non-numeric columns with mode
])


### ENCODING PIPELINE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function wrapped in a custom transformer
class AggregateLowCounts(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=100):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Loop through categorical columns only
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            # Use value_counts and map efficiently to avoid multiple passes
            value_counts = X[column].value_counts()
            
            # Find categories to replace with 'Other'
            low_count_categories = value_counts[value_counts <= self.threshold].index
            
            # Replace low count categories with 'Other'
            X[column] = X[column].where(~X[column].isin(low_count_categories), 'Other')
        
        return X

# Custom transformer for binary encoding
class EncodeBinaryVariables(BaseEstimator, TransformerMixin):
    def __init__(self, binary_columns):
        self.binary_columns = binary_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        label_encoder = LabelEncoder()
        
        for col in self.binary_columns:
            if col in X.columns:  # Check if column exists
                X[col] = label_encoder.fit_transform(X[col])
        
        return X

# Custom transformer for ordinal encoding
class EncodeOrdinalVariables(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_column, mapping_dict, default_value=None):
        self.ordinal_column = ordinal_column
        self.mapping_dict = mapping_dict
        self.default_value = default_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.ordinal_column in X.columns:  # Check if column exists
            X[self.ordinal_column] = X[self.ordinal_column].map(self.mapping_dict)
            if self.default_value is not None:
                X[self.ordinal_column] = X[self.ordinal_column].fillna(self.default_value)
        
        return X

# Custom transformer for nominal encoding
class EncodeNominalVariables(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_columns, drop_first=True):
        self.nominal_columns = nominal_columns
        self.drop_first = drop_first
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Only apply get_dummies if the columns are still in the DataFrame
        for col in self.nominal_columns:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col], drop_first=self.drop_first)
        
        return X

# Define columns for encoding
binary_columns = ['CODE_GENDER', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR']
ordinal_mapping = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}
nominal_columns = [
    'NAME_TYPE_SUITE', 
    'NAME_INCOME_TYPE', 
    'NAME_FAMILY_STATUS', 
    'NAME_HOUSING_TYPE', 
    'ORGANIZATION_TYPE', 
    'OCCUPATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START',
    'FONDKAPREMONT_MODE',        
    'HOUSETYPE_MODE',            
    'WALLSMATERIAL_MODE',        
    'EMERGENCYSTATE_MODE'        
]

# Create the pipeline with the new aggregation step
application_encoding_pipeline = Pipeline(steps=[
    ('aggregate_low_counts', AggregateLowCounts(threshold=100)),
    ('binary_encoder', EncodeBinaryVariables(binary_columns=binary_columns)),
    ('ordinal_encoder', EncodeOrdinalVariables(ordinal_column='NAME_EDUCATION_TYPE', mapping_dict=ordinal_mapping, default_value=0)),
    ('nominal_encoder', EncodeNominalVariables(nominal_columns=nominal_columns))
])



### APPLICATION_REDUCTION_PIPELINE

# Custom transformer to select top features
class TopFeatureSelector:
    def __init__(self, top_features):
        self.top_features = top_features

    def transform(self, X):
        # Check if TARGET is in top_features and also in X columns
        valid_features = [feat for feat in self.top_features if feat in X.columns]
        return X[valid_features]



# Custom transformer to create aggregated indexes
class Aggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Create a new DataFrame with aggregated columns
        aggregated_df = pd.DataFrame(index=X.index)
        
        # Aggregating Document Flags
        aggregated_df['TOTAL_DOCUMENT_FLAGS'] = X[[col for col in X.columns if col.startswith("FLAG_DOCUMENT")]].sum(axis=1)
        
        # Creating a Stability Index
        aggregated_df['STABILITY_INDEX'] = X[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 
                                              'REG_CITY_NOT_WORK_CITY', 'REG_CITY_NOT_LIVE_CITY']].mean(axis=1)
        
        
        # Creating a Contact Index
        aggregated_df['CONTACT_INDEX'] = X[['FLAG_PHONE', 'FLAG_EMAIL', 'FLAG_EMP_PHONE']].sum(axis=1)
        
    
        # Aggregating Credit Bureau Request Totals
        aggregated_df['CREDIT_BUREAU_REQ_TOTAL'] = X[['AMT_REQ_CREDIT_BUREAU_HOUR', 
                                                      'AMT_REQ_CREDIT_BUREAU_DAY', 
                                                      'AMT_REQ_CREDIT_BUREAU_WEEK', 
                                                      'AMT_REQ_CREDIT_BUREAU_MON', 
                                                      'AMT_REQ_CREDIT_BUREAU_QRT', 
                                                      'AMT_REQ_CREDIT_BUREAU_YEAR']].sum(axis=1)

        # Aggregating Social Circle Observations
        aggregated_df['SOCIAL_CIRCLE_OBS_TOTAL'] = X[['OBS_30_CNT_SOCIAL_CIRCLE', 
                                                      'OBS_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)
        
        # Aggregating Social Circle Defaults
        aggregated_df['SOCIAL_CIRCLE_DEF_TOTAL'] = X[['DEF_30_CNT_SOCIAL_CIRCLE', 
                                                      'DEF_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)
        
        return aggregated_df
    
# Custom transformer to convert time-based features to absolute values
class MakeAbsolute(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original data
        X[self.columns] = X[self.columns].abs()  # Apply absolute transformation
        return X


# Define top features list
top_features_list = [
    "TARGET", "SK_ID_CURR", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION",
    "AMT_ANNUITY", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE", "AMT_CREDIT", 
    "REGION_POPULATION_RELATIVE", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "EXT_SOURCE_1", 
    "HOUR_APPR_PROCESS_START", "AMT_REQ_CREDIT_BUREAU_YEAR", 
    "OWN_CAR_AGE", "LIVINGAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "OBS_30_CNT_SOCIAL_CIRCLE", 
    "APARTMENTS_AVG", "OBS_60_CNT_SOCIAL_CIRCLE", 
    "CNT_FAM_MEMBERS", "COMMONAREA_AVG", "YEARS_BUILD_AVG", "NAME_EDUCATION_TYPE", "NONLIVINGAREA_AVG", 
    "LANDAREA_AVG", "BASEMENTAREA_AVG", "LIVINGAPARTMENTS_AVG", "CNT_CHILDREN", "NAME_FAMILY_STATUS_Married", "CODE_GENDER", "FLAG_OWN_REALTY"
]



# Creating the pipeline

application_reduction_pipeline = Pipeline([
    ('top_feature_selector', TopFeatureSelector(top_features=top_features_list)),
    ('aggregator', Aggregator())]
)













###CREDIT AGGREGATOR:
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# Define the aggregation configuration as a dictionary
aggregation_config = {
    'SK_ID_PREV': 'count',
    'MONTHS_BALANCE': 'max',
    'AMT_BALANCE': ['sum', 'mean'],
    'AMT_CREDIT_LIMIT_ACTUAL': 'mean',
    'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'mean'],
    'AMT_DRAWINGS_CURRENT': ['sum', 'mean'],
    'AMT_DRAWINGS_OTHER_CURRENT': ['sum', 'mean'],
    'AMT_DRAWINGS_POS_CURRENT': ['sum', 'mean'],
    'AMT_INST_MIN_REGULARITY': 'mean',
    'AMT_PAYMENT_CURRENT': ['sum', 'mean'],
    'AMT_PAYMENT_TOTAL_CURRENT': ['sum', 'mean'],
    'AMT_RECEIVABLE_PRINCIPAL': ['sum', 'mean'],
    'AMT_RECIVABLE': ['sum', 'mean'],
    'AMT_TOTAL_RECEIVABLE': ['sum', 'mean'],
    'CNT_DRAWINGS_ATM_CURRENT': 'sum',
    'CNT_DRAWINGS_CURRENT': 'sum',
    'CNT_DRAWINGS_OTHER_CURRENT': 'sum',
    'CNT_DRAWINGS_POS_CURRENT': 'sum',
    'CNT_INSTALMENT_MATURE_CUM': 'sum',
    'NAME_CONTRACT_STATUS': lambda x: x.mode()[0],
    'SK_DPD': 'max',
    'SK_DPD_DEF': 'max'
}

# Custom transformer for aggregation
class AggregationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, group_key, aggregation_config):
        """
        Initializes the AggregationTransformer with the group key and aggregation configuration.

        Parameters:
        - group_key: The column to group by (e.g., 'SK_ID_CURR')
        - aggregation_config: A dictionary where keys are column names and values are aggregation functions
        """
        
        self.group_key = group_key  # Define the column to group by
        self.aggregation_config = aggregation_config  # Customizable aggregation configuration

    def fit(self, X, y=None):
        return self  # No fitting needed for aggregation

    def transform(self, X):
        # Perform aggregation based on the passed configuration
        aggregated_df = X.groupby(self.group_key).agg(self.aggregation_config).reset_index()

        # Flatten the column names if there are multiple aggregations for a single column
        aggregated_df.columns = [
            col[0] if col[0] == self.group_key else '_'.join(col).strip()
            for col in aggregated_df.columns
        ]

        return aggregated_df



# Create the pipeline with the flexible configuration
credit_aggregation_pipeline = Pipeline([
    ('aggregator', AggregationTransformer(group_key='SK_ID_CURR', aggregation_config=aggregation_config))
])
















### CREDIT CLEANER

# FlagMissing transformer 
class CFlagMissing(BaseEstimator, TransformerMixin):
    """
    Flags missing values by creating new columns for each feature indicating if a value is missing.
    This helps track where data was missing for future analysis.
    """
    def __init__(self, columns_to_flag):
        self.columns_to_flag = columns_to_flag
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Avoid modifying original data during transformation
        
        # Create a separate DataFrame for the flags
        missing_flags = pd.DataFrame()
        
        for col in self.columns_to_flag:
            missing_flags[f'FLAG_{col}'] = np.where(X[col].isnull(), 0, 1)
        
        # Concatenate the flags to the original DataFrame (this is done to avoid fragmentation)
        X = pd.concat([X, missing_flags], axis=1)
        
        return X

# Custom transformer to impute missing values with specified strategies
class CImputeColumns(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using different strategies:
    - Median for selected columns.
    - 'Unknown' for selected categorical columns.
    - Zero for other selected columns.
    """
    def __init__(self, unknown_columns=None):
        self.unknown_columns = unknown_columns
  
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Impute with 'Unknown' for specified categorical columns
        if self.unknown_columns:
            for col in self.unknown_columns:
                X[col].fillna('Unknown', inplace=True)
        
        return X

# Custom transformer to cap outliers before scaling, ensuring percentiles are only calculated once
class CCapOutliers(BaseEstimator, TransformerMixin):
    """
    Caps outliers for numerical columns by setting values above the specified percentiles.
    The percentiles are calculated once during the fit and applied consistently during transforms.
    """
    def __init__(self, columns_to_cap, lower_percentile=2, upper_percentile=98):
        self.columns_to_cap = columns_to_cap
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_caps_ = {}
        self.upper_caps_ = {}
    
    def fit(self, X, y=None):
        # Calculate and store the caps for each column during fit
        for col in self.columns_to_cap:
            self.lower_caps_[col] = X[col].quantile(self.lower_percentile / 100)
            self.upper_caps_[col] = X[col].quantile(self.upper_percentile / 100)
        return self
    
    def transform(self, X):
        X = X.copy()  # Work on a copy to avoid modifying the original dataframe
        # Apply the pre-calculated caps during transform
        for col in self.columns_to_cap:
            X[col] = X[col].clip(lower=self.lower_caps_[col], upper=self.upper_caps_[col])
        return X

# Custom transformer to scale numerical columns
class CScaleNumericFeatures(BaseEstimator, TransformerMixin):
    """
    Scales numerical features using StandardScaler.
    This ensures that all features are on a similar scale, which helps models converge faster.
    """
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X

# Custom transformer to impute remaining missing values with mean for numeric columns
class CImputeWithMedian(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for numeric columns using the mean of each column.
    This is useful for handling the last few missing values in numeric columns.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        return X

# Custom transformer to impute remaining missing values with mode for non-numeric columns
class CImputeWithMode(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for non-numeric columns using the most frequent value (mode).
    This is useful for categorical data.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(exclude=['float64', 'int64']).columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)
        return X
    

credit_flag_cols= [
    'AMT_PAYMENT_CURRENT_mean',
    'AMT_DRAWINGS_POS_CURRENT_mean',
    'AMT_DRAWINGS_OTHER_CURRENT_mean',
    'AMT_DRAWINGS_ATM_CURRENT_mean'
]
credit_numeric_cols = ['SK_ID_CURR', 'SK_ID_PREV_count', 'MONTHS_BALANCE_max', 'AMT_BALANCE_sum', 'AMT_BALANCE_mean', 'AMT_CREDIT_LIMIT_ACTUAL_mean', 'AMT_DRAWINGS_ATM_CURRENT_sum', 'AMT_DRAWINGS_ATM_CURRENT_mean', 'AMT_DRAWINGS_CURRENT_sum', 'AMT_DRAWINGS_CURRENT_mean', 'AMT_DRAWINGS_OTHER_CURRENT_sum', 'AMT_DRAWINGS_OTHER_CURRENT_mean', 'AMT_DRAWINGS_POS_CURRENT_sum', 'AMT_DRAWINGS_POS_CURRENT_mean', 'AMT_INST_MIN_REGULARITY_mean', 'AMT_PAYMENT_CURRENT_sum', 'AMT_PAYMENT_CURRENT_mean', 'AMT_PAYMENT_TOTAL_CURRENT_sum', 'AMT_PAYMENT_TOTAL_CURRENT_mean', 'AMT_RECEIVABLE_PRINCIPAL_sum', 'AMT_RECEIVABLE_PRINCIPAL_mean', 'AMT_RECIVABLE_sum', 'AMT_RECIVABLE_mean', 'AMT_TOTAL_RECEIVABLE_sum', 'AMT_TOTAL_RECEIVABLE_mean', 'CNT_DRAWINGS_ATM_CURRENT_sum', 'CNT_DRAWINGS_CURRENT_sum', 'CNT_DRAWINGS_OTHER_CURRENT_sum', 'CNT_DRAWINGS_POS_CURRENT_sum', 'CNT_INSTALMENT_MATURE_CUM_sum', 'SK_DPD_max', 'SK_DPD_DEF_max']
# Create the pipeline that chains all the transformations together
credit_cleaning_pipeline = Pipeline(steps=[
    ('flag_missing', CFlagMissing(columns_to_flag=credit_flag_cols)),
    ('cap_outliers', CCapOutliers(columns_to_cap=credit_numeric_cols)),  # Cap outliers
    ('mean_impute', CImputeWithMedian()),   # Impute remaining numeric columns with median
    ('mode_impute', CImputeWithMode())    # Impute remaining non-numeric columns with mode
])


### CREDIT ENCODING

# Custom transformer for binary encoding
class CEncodeBinaryVariables(BaseEstimator, TransformerMixin):
    def __init__(self, binary_columns):
        self.binary_columns = binary_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        label_encoder = LabelEncoder()
        
        for col in self.binary_columns:
            if col in X.columns:  # Check if column exists
                X[col] = label_encoder.fit_transform(X[col])
        
        return X


# Custom transformer for nominal encoding
class CEncodeNominalVariables(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_columns, drop_first=True):
        self.nominal_columns = nominal_columns
        self.drop_first = drop_first
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Only apply get_dummies if the columns are still in the DataFrame
        for col in self.nominal_columns:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col], drop_first=self.drop_first)
        
        return X


Cbinary_columns = ["FLAG_AMT_CURRENT_PAYMENT_mean", "FLAG_AMT_DRAWING_POS_mean", "FLAG_AMT_DRAWINGS_OTHER_CURRENT_mean", "FLAG_AMT_DRAWINGS_ATM_CURRENT_mean"]
Cnominal_columns = ["NAME_CONTRACT_STATUS_<lambda>"]
# Create the pipeline
credit_encoding_pipeline = Pipeline(steps=[
    ('binary_encoder', CEncodeBinaryVariables(binary_columns=Cbinary_columns)),
    ('nominal_encoder', CEncodeNominalVariables(nominal_columns=Cnominal_columns))
])



### REDUCTION:
# Time features specific to credit dataset
credit_time_features = [
    'MONTHS_BALANCE_max'
]


Cselected_features = [
    'SK_ID_CURR', 'AMT_PAYMENT_CURRENT_mean', 'AMT_DRAWINGS_OTHER_CURRENT_sum', 
    'CNT_DRAWINGS_ATM_CURRENT_sum', 'AMT_TOTAL_RECEIVABLE_mean', 
    'AMT_BALANCE_mean', 'AMT_CREDIT_LIMIT_ACTUAL_mean', 'SK_ID_PREV_count', 
    'AMT_INST_MIN_REGULARITY_mean', 'AMT_RECIVABLE_mean', 
    'AMT_RECEIVABLE_PRINCIPAL_mean', 'AMT_RECIVABLE_sum', 
    'AMT_RECEIVABLE_PRINCIPAL_sum', 'CNT_INSTALMENT_MATURE_CUM_sum', 
    'MONTHS_BALANCE_max', 'SK_DPD_max', 'SK_DPD_DEF_max'
]

# Define the `Payment_Behavior_Index` columns
Cpayment_behavior_flags = [
    'FLAG_AMT_PAYMENT_CURRENT_mean', 
    'FLAG_AMT_DRAWINGS_POS_CURRENT_mean', 
    'FLAG_AMT_DRAWINGS_OTHER_CURRENT_mean', 
    'FLAG_AMT_DRAWINGS_ATM_CURRENT_mean'
]

# Custom transformer for selecting features
class CFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features]

# Custom transformer to create `Payment_Behavior_Index`
class CPaymentBehaviorAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, flags):
        self.flags = flags
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Create `Payment_Behavior_Index` as the mean of the specified flag columns
        X = X.copy()
        X['Payment_Behavior_Index'] = X[self.flags].mean(axis=1)
        # Return only the `Payment_Behavior_Index`
        return X[['Payment_Behavior_Index']]

# Custom transformer to convert time-based features to absolute values
class MakeAbsolute(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original data
        X[self.columns] = X[self.columns].abs()  # Apply absolute transformation
        return X

# Define the updated credit reduction pipeline
credit_reduction_pipeline = Pipeline([
    ('feature_selector', CFeatureSelector(features=Cselected_features)),
    ('payment_behavior_aggregator', CPaymentBehaviorAggregator(flags=Cpayment_behavior_flags))
])


### FEATURE PIPELINES
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

###funcs
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Scaling function
def scale_columns(df, columns_to_scale, scaler=StandardScaler()):
    df = df.copy()  # Avoid mutating the original DataFrame
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df, scaler  # Return scaler to apply it on test set

# Function to create non-linear income features
def non_linear_income_features(df):
    """
    Create income quartile-based features for default risk prediction.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame to add features to. Requires columns for AMT_INCOME_TOTAL, AMT_CREDIT, etc.
    
    Returns:
    - df: pd.DataFrame - DataFrame with new income quartile-based features added.
    """
    # Create income quartile feature
    df['AMT_INCOME_QUARTILE'] = pd.qcut(df['AMT_INCOME_TOTAL'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['AMT_INCOME_QUARTILE'] = df['AMT_INCOME_QUARTILE'].map({'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}).astype(float)

    # Interaction and derived features
    df['AMT_INCOME_TOTAL_squared'] = df['AMT_INCOME_TOTAL'] ** 2
    df['INCOME_BIN_CREDIT'] = df['AMT_INCOME_QUARTILE'] * df['AMT_CREDIT']
    df['INCOME_QUARTILE_ANNUITY_RATIO'] = df['AMT_INCOME_QUARTILE'] * df['AMT_ANNUITY']
    df['INCOME_QUARTILE_EMPLOYED_RATIO'] = df['AMT_INCOME_QUARTILE'] * df['DAYS_EMPLOYED']
    df['INCOME_QUARTILE_EXT_SOURCE2'] = df['AMT_INCOME_QUARTILE'] * df['EXT_SOURCE_2']
    df['INCOME_QUARTILE_EXT_SOURCE3'] = df['AMT_INCOME_QUARTILE'] * df['EXT_SOURCE_3']
    df['INCOME_QUARTILE_AGE'] = df['AMT_INCOME_QUARTILE'] * abs(df['DAYS_BIRTH'])
    df['INCOME_QUARTILE_DEBT_RATIO'] = df['AMT_INCOME_QUARTILE'] * (df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5))
    df['INCOME_CREDIT_HISTORY_STABILITY'] = df['AMT_INCOME_QUARTILE'] * (df['SK_DPD_max'] + df['SK_DPD_DEF_max'])
    df['INCOME_DEBT_BURDEN_INDEX'] = df['AMT_INCOME_QUARTILE'] * (df['AMT_ANNUITY'] + df['AMT_TOTAL_RECEIVABLE_mean']) / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['DEBT_TO_CREDIT_QUARTILE'] = df['AMT_INCOME_QUARTILE'] * (df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1e-5))

    return df

# Function to create wealth-related features
def create_wealth_features(df):
    """
    Creates new wealth-related features directly on the provided DataFrame.
    """
    # Create new features directly in the input DataFrame
    quartile_threshold = df['AMT_GOODS_PRICE'].quantile(0.25)
    df['binary_feature'] = (df['AMT_GOODS_PRICE'] <= quartile_threshold).astype(int)
    df['income_annuity_ratio'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    df['income_days_birth_ratio'] = df['AMT_INCOME_TOTAL'] / (-df['DAYS_BIRTH'])
    df['days_employed_days_birth_ratio'] = df['DAYS_EMPLOYED'] / (-df['DAYS_BIRTH'])
    df['current_payments_income_ratio'] = df['AMT_PAYMENT_CURRENT_mean'] / df['AMT_INCOME_TOTAL']
    df['days_employed_income_product'] = df['DAYS_EMPLOYED'] * df['AMT_INCOME_TOTAL']
    df['financial_flexibility_score'] = (df['AMT_INCOME_TOTAL'] - df['AMT_PAYMENT_CURRENT_mean']) / df['AMT_GOODS_PRICE']
    df['dependency_load_index'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    return df

# Function to create overextension-related features
def overextension_features(df):
    """
    Adds overextension-related features to the given DataFrame in place.
    
    Parameters:
    df (pd.DataFrame): DataFrame to modify with overextension features.

    Returns:
    pd.DataFrame: The modified DataFrame with new features added.
    """
    # 1. Annuity-to-Income Ratio
    df['annuity_income_ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # 2. Credit-to-Income Ratio
    df['credit_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # 3. Annuity-to-Credit Ratio
    df['annuity_credit_ratio'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # 4. Current Payments Mean to Income Ratio
    df['current_payments_income_ratio'] = df['AMT_PAYMENT_CURRENT_mean'] / df['AMT_INCOME_TOTAL']
    
    # 5. Credit Utilization Ratio
    df['credit_utilization_ratio'] = df['AMT_BALANCE_mean'] / df['AMT_CREDIT_LIMIT_ACTUAL_mean']
    
    # 6. Debt-to-Asset Ratio
    df['debt_asset_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    return df



# Custom transformer for scaling columns
class ScaleColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X

# Custom transformer for non-linear income features
class NonLinearIncomeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return non_linear_income_features(X)

# Custom transformer for wealth-related features
class WealthFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return create_wealth_features(X)

# Custom transformer for overextension-related features
class OverextensionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return overextension_features(X)
    
class SetIndex(BaseEstimator, TransformerMixin):
    def __init__(self, index_column):
        self.index_column = index_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.set_index(self.index_column, inplace=True)
        return X

# Define the feature engineering pipeline
def create_feature_engineering_pipeline(columns_to_scale, index_column="SK_ID_CURR"):
    feature_pipeline = Pipeline([
        ('non_linear_income_features', NonLinearIncomeFeatures()),
        ('wealth_features', WealthFeatures()),
        ('overextension_features', OverextensionFeatures()),
        ('scaling', ScaleColumns(columns_to_scale=columns_to_scale)),
        ('set_index', SetIndex(index_column=index_column))
        
    ])
    return feature_pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define default values for columns to scale and RFE-selected features
all_columns_to_scale = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'AMT_ANNUITY', 
    'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT', 'REGION_POPULATION_RELATIVE', 'AMT_INCOME_TOTAL', 
    'AMT_GOODS_PRICE', 'EXT_SOURCE_1', 'HOUR_APPR_PROCESS_START', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'OWN_CAR_AGE', 
    'LIVINGAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'OBS_30_CNT_SOCIAL_CIRCLE', 'APARTMENTS_AVG', 
    'OBS_60_CNT_SOCIAL_CIRCLE', 'CNT_FAM_MEMBERS', 'COMMONAREA_AVG', 'YEARS_BUILD_AVG', 'NONLIVINGAREA_AVG', 
    'LANDAREA_AVG', 'BASEMENTAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'CNT_CHILDREN', 'TOTAL_DOCUMENT_FLAGS', 
    'STABILITY_INDEX', 'CONTACT_INDEX', 'CREDIT_BUREAU_REQ_TOTAL', 'SOCIAL_CIRCLE_OBS_TOTAL', 'SOCIAL_CIRCLE_DEF_TOTAL', 
    'AMT_PAYMENT_CURRENT_mean', 'AMT_DRAWINGS_OTHER_CURRENT_sum', 'CNT_DRAWINGS_ATM_CURRENT_sum', 'AMT_TOTAL_RECEIVABLE_mean', 
    'AMT_BALANCE_mean', 'AMT_CREDIT_LIMIT_ACTUAL_mean', 'SK_ID_PREV_count', 'AMT_INST_MIN_REGULARITY_mean', 'AMT_RECIVABLE_mean', 
    'AMT_RECEIVABLE_PRINCIPAL_mean', 'AMT_RECIVABLE_sum', 'AMT_RECEIVABLE_PRINCIPAL_sum', 'CNT_INSTALMENT_MATURE_CUM_sum', 
    'MONTHS_BALANCE_max', 'SK_DPD_max', 'SK_DPD_DEF_max', 'Payment_Behavior_Index', 'AMT_INCOME_TOTAL_squared', 
    'INCOME_BIN_CREDIT', 'INCOME_QUARTILE_ANNUITY_RATIO', 'INCOME_QUARTILE_EMPLOYED_RATIO', 'INCOME_QUARTILE_EXT_SOURCE2', 
    'INCOME_QUARTILE_EXT_SOURCE3', 'INCOME_QUARTILE_AGE', 'INCOME_QUARTILE_DEBT_RATIO', 'INCOME_CREDIT_HISTORY_STABILITY', 
    'INCOME_DEBT_BURDEN_INDEX', 'DEBT_TO_CREDIT_QUARTILE', 'Payment_Behavior_Index', 'financial_flexibility_score', 
    'dependency_load_index', 'days_employed_days_birth_ratio', 'income_annuity_ratio', 'income_days_birth_ratio', 
    'current_payments_income_ratio', 'days_employed_income_product', 'annuity_income_ratio', 'credit_income_ratio', 
    'annuity_credit_ratio', 'current_payments_income_ratio', 'debt_asset_ratio', 'credit_utilization_ratio'
]

rfe_selected_features = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 
    'EXT_SOURCE_1', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS_Married', 'CODE_GENDER', 'STABILITY_INDEX', 'CREDIT_BUREAU_REQ_TOTAL', 
    'CNT_DRAWINGS_ATM_CURRENT_sum', 'AMT_TOTAL_RECEIVABLE_mean', 'AMT_BALANCE_mean', 'AMT_CREDIT_LIMIT_ACTUAL_mean', 
    'AMT_INST_MIN_REGULARITY_mean', 'AMT_RECIVABLE_mean', 'AMT_RECEIVABLE_PRINCIPAL_mean', 'AMT_RECIVABLE_sum', 
    'AMT_RECEIVABLE_PRINCIPAL_sum', 'CNT_INSTALMENT_MATURE_CUM_sum', 'AMT_INCOME_QUARTILE', 'AMT_INCOME_TOTAL_squared', 
    'INCOME_BIN_CREDIT', 'INCOME_QUARTILE_ANNUITY_RATIO', 'INCOME_QUARTILE_EXT_SOURCE2', 'INCOME_QUARTILE_EXT_SOURCE3', 
    'INCOME_QUARTILE_AGE', 'DEBT_TO_CREDIT_QUARTILE', 'binary_feature', 'income_days_birth_ratio', 
    'days_employed_days_birth_ratio', 'credit_income_ratio', 'annuity_credit_ratio', 'credit_utilization_ratio', 'debt_asset_ratio', "TARGET"
]

# Custom transformer for selecting RFE-selected features
class RFEFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features=rfe_selected_features, include_target=True):
        self.selected_features = selected_features
        self.include_target = include_target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Exclude 'TARGET' if not needed
        features_to_select = [feat for feat in self.selected_features if feat in X.columns]
        if not self.include_target:
            features_to_select = [feat for feat in features_to_select if feat != "TARGET"]
        return X[features_to_select]

# Define the feature engineering pipeline with RFE feature selection
def create_feature_engineering_pipeline_rfe(include_target=True):
    feature_pipeline_rfe = Pipeline([
        ('non_linear_income_features', NonLinearIncomeFeatures()),
        ('wealth_features', WealthFeatures()),
        ('overextension_features', OverextensionFeatures()),
        ('scaling', ScaleColumns(columns_to_scale=all_columns_to_scale)),
        ('set_index', SetIndex(index_column="SK_ID_CURR")),
        ('feature_selection', RFEFeatureSelector(include_target=include_target))  # Pass the `include_target` parameter
    ])
    return feature_pipeline_rfe


### function to merge application data with credit data in chunks
def merge_application_with_credit(application_df, credit_df, chunk_size=10000):
    merged_result = pd.DataFrame()
    for i in range(0, len(application_df), chunk_size):
        # Process each chunk and merge with credit data
        chunk = application_df.iloc[i:i+chunk_size]
        merged_chunk = pd.merge(chunk, credit_df, on='SK_ID_CURR', how='inner')
        merged_result = pd.concat([merged_result, merged_chunk], ignore_index=True)
    return merged_result
















#### UNIFIED PIPELINE

import pandas as pd
from pipelines import (
    application_cleaning_pipeline,
    application_encoding_pipeline,
    application_reduction_pipeline,
    credit_aggregation_pipeline,
    credit_cleaning_pipeline,
    credit_encoding_pipeline,
    credit_reduction_pipeline,
    create_feature_engineering_pipeline_rfe
)

def unified_pipeline(application_df, credit_df, chunk_size=10000, include_target=True):
    """
    A unified pipeline to process application_test data by cleaning, encoding, reducing features,
    merging with credit data, and applying feature engineering.
    """
    # Step 1: Application Data Cleaning and Encoding
    application_df = application_cleaning_pipeline.fit_transform(application_df)
    application_df = application_encoding_pipeline.fit_transform(application_df)
    
    # Step 2: Application Data Reduction (Feature Selection and Aggregation)
    application_df = pd.concat([
        application_reduction_pipeline.named_steps['top_feature_selector'].transform(application_df),
        application_reduction_pipeline.named_steps['aggregator'].transform(application_df)
    ], axis=1)
    
    # Step 3: Credit Data Processing (Cleaning, Encoding, and Aggregation)
    credit_df = credit_aggregation_pipeline.fit_transform(credit_df)
    credit_df = credit_cleaning_pipeline.fit_transform(credit_df)
    credit_df = credit_encoding_pipeline.fit_transform(credit_df)
    credit_df = pd.concat([
        credit_reduction_pipeline.named_steps['feature_selector'].transform(credit_df),
        credit_reduction_pipeline.named_steps['payment_behavior_aggregator'].transform(credit_df)
    ], axis=1)
    
    # Step 4: Merge Application and Credit Data in Chunks
    def merge_application_with_credit(application_df, credit_df, chunk_size=10000):
        merged_result = pd.DataFrame()
        for i in range(0, len(application_df), chunk_size):
            chunk = application_df.iloc[i:i+chunk_size]
            merged_chunk = pd.merge(chunk, credit_df, on='SK_ID_CURR', how='inner')
            merged_result = pd.concat([merged_result, merged_chunk], ignore_index=True)
        return merged_result
    
    merged_df = merge_application_with_credit(application_df, credit_df, chunk_size=chunk_size)
    
    # Step 5: Feature Engineering with RFE
    FE_pipeline = create_feature_engineering_pipeline_rfe(include_target=include_target)
    FE_merged_df = FE_pipeline.fit_transform(merged_df)
    
    return FE_merged_df

