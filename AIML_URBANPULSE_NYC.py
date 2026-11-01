

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"



df = pd.read_csv(url + "?$limit=10000")



print("Dataset loaded successfully")



print("Rows and Columns:", df.shape)



print(df.head())


# Display first 10 records

print(df.head(10))


# Display last 5 records

print(df.tail())


# Display column names

print(df.columns.tolist())


# Display data types

print(df.dtypes)


# Display summary statistics

print(df.describe(include='all'))


# Display dataset information

df.info()


missing_values = df.isnull().sum()



print(missing_values)


missing_percentage = (

    df.isnull().sum() / len(df)

) * 100



missing_report = pd.DataFrame({

    'Missing Count': df.isnull().sum(),

    'Missing Percentage': missing_percentage

})



print(

    missing_report.sort_values(

        by='Missing Count',

        ascending=False

    )

)


plt.figure(figsize=(12, 6))



missing_report['Missing Percentage'].sort_values(

    ascending=False

).head(20).plot(kind='bar')



plt.title("Top 20 Columns with Missing Values")

plt.xlabel("Columns")

plt.ylabel("Missing Percentage")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()


# Create a working copy

df_clean = df.copy()



# Fill missing categorical values

df_clean['borough'] = (

    df_clean['borough'].fillna('UNKNOWN')

)



df_clean['descriptor'] = (

    df_clean['descriptor'].fillna('Not Provided')

)



df_clean['status'] = (

    df_clean['status'].fillna('Unknown')

)



# Check remaining missing values

print(df_clean.isnull().sum())


duplicate_count = df_clean.duplicated().sum()



print("Duplicate Rows:", duplicate_count)



df_clean = df_clean.drop_duplicates()



print("Rows after removing duplicates:", len(df_clean))


print(

    "Duplicate Complaint IDs:",

    df_clean['unique_key'].duplicated().sum()

)


df_clean['created_date'] = pd.to_datetime(

    df_clean['created_date'],

    errors='coerce'

)



df_clean['closed_date'] = pd.to_datetime(

    df_clean['closed_date'],

    errors='coerce'

)



print(df_clean[['created_date', 'closed_date']].dtypes)


df_clean['resolution_hours'] = (

    df_clean['closed_date'] -

    df_clean['created_date']

).dt.total_seconds() / 3600



# Remove invalid negative durations

df_clean.loc[

    df_clean['resolution_hours'] < 0,

    'resolution_hours'

] = np.nan



print(df_clean['resolution_hours'].describe())


df_clean.to_csv(

    "nyc_311_cleaned.csv",

    index=False

)



print("Cleaned dataset saved successfully!")


print(df_clean.shape)

print(df_clean.head())

print(df_clean.isnull().sum())


with open("analysis_output.txt", "w") as f:

    f.write("Dataset Shape:\n")

    f.write(str(df_clean.shape))



    f.write("\n\nFirst 5 Rows:\n")

    f.write(df_clean.head().to_string())



    f.write("\n\nMissing Values:\n")

    f.write(df_clean.isnull().sum().to_string())






import pandas as pd

import numpy as np



# Create a copy

df_clean = df.copy()



# Remove exact duplicates

df_clean = df_clean.drop_duplicates()



# Fill missing categorical values

df_clean['borough'] = df_clean['borough'].fillna('UNKNOWN')

df_clean['descriptor'] = df_clean['descriptor'].fillna('Not Provided')

df_clean['location_type'] = df_clean['location_type'].fillna('Unknown')



# Convert dates

df_clean['created_date'] = pd.to_datetime(

    df_clean['created_date'], errors='coerce'

)



df_clean['closed_date'] = pd.to_datetime(

    df_clean['closed_date'], errors='coerce'

)



# Calculate resolution time

df_clean['resolution_hours'] = (

    df_clean['closed_date'] - df_clean['created_date']

).dt.total_seconds() / 3600



df_clean.loc[

    df_clean['resolution_hours'] < 0,

    'resolution_hours'

] = np.nan



print("Cleaned Dataset:", df_clean.shape)

print("Duplicate IDs:", df_clean['unique_key'].duplicated().sum())


print(df_clean['complaint_type'].value_counts().head(10))


print(df_clean['borough'].value_counts())


print(df_clean['status'].value_counts())


import matplotlib.pyplot as plt

import seaborn as sns



top10 = df_clean['complaint_type'].value_counts().head(10)



plt.figure(figsize=(12, 6))



sns.barplot(

    x=top10.values,

    y=top10.index

)



plt.title("Top 10 NYC 311 Complaint Types")

plt.xlabel("Complaint Count")

plt.ylabel("Complaint Type")

plt.tight_layout()

plt.show()


borough_counts = df_clean['borough'].value_counts()



print(borough_counts)



plt.figure(figsize=(10, 5))



sns.barplot(

    x=borough_counts.index,

    y=borough_counts.values

)



plt.title("NYC 311 Complaints by Borough")

plt.xlabel("Borough")

plt.ylabel("Number of Complaints")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


status_counts = df_clean['status'].value_counts()



print(status_counts)



plt.figure(figsize=(9, 5))



sns.barplot(

    x=status_counts.index,

    y=status_counts.values

)



plt.title("NYC 311 Complaint Status")

plt.xlabel("Status")

plt.ylabel("Complaint Count")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


top10 = df_clean['complaint_type'].value_counts().head(10)



plt.figure(figsize=(12, 6))



ax = sns.barplot(

    x=top10.values,

    y=top10.index

)



for container in ax.containers:

    ax.bar_label(container, padding=5)



plt.title("Top 10 NYC 311 Complaint Types")

plt.xlabel("Number of Complaints")

plt.ylabel("Complaint Type")

plt.tight_layout()

plt.show()


borough_counts = df_clean['borough'].value_counts()



print(borough_counts)



plt.figure(figsize=(10, 6))



ax = sns.barplot(

    x=borough_counts.index,

    y=borough_counts.values

)



for container in ax.containers:

    ax.bar_label(container, padding=3)



plt.title("NYC 311 Complaints by Borough")

plt.xlabel("Borough")

plt.ylabel("Complaint Count")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# Only include closed complaints with valid resolution times

resolved_df = df_clean[

    (df_clean['status'] == 'Closed') &

    (df_clean['resolution_hours'].notna())

].copy()



borough_resolution = (

    resolved_df.groupby('borough')['resolution_hours']

    .mean()

    .sort_values(ascending=False)

)



print(borough_resolution.round(2))



plt.figure(figsize=(10, 6))



ax = sns.barplot(

    x=borough_resolution.index,

    y=borough_resolution.values

)



for container in ax.containers:

    ax.bar_label(container, fmt='%.1f', padding=3)



plt.title("Average Complaint Resolution Time by Borough")

plt.xlabel("Borough")

plt.ylabel("Average Resolution Time (Hours)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


agency_counts = (

    df_clean['agency_name']

    .value_counts()

    .head(10)

)



print(agency_counts)



plt.figure(figsize=(12, 6))



ax = sns.barplot(

    x=agency_counts.values,

    y=agency_counts.index

)



for container in ax.containers:

    ax.bar_label(container, padding=4)



plt.title("Top 10 NYC Agencies by Complaint Count")

plt.xlabel("Number of Complaints")

plt.ylabel("Agency")

plt.tight_layout()

plt.show()


agency_resolution = (

    resolved_df.groupby('agency_name')['resolution_hours']

    .agg(['count', 'mean'])

)



# Include agencies with at least 30 closed complaints

agency_resolution = agency_resolution[

    agency_resolution['count'] >= 30

]



agency_resolution = (

    agency_resolution.sort_values(

        'mean', ascending=False

    ).head(10)

)



print(agency_resolution.round(2))



plt.figure(figsize=(12, 6))



ax = sns.barplot(

    x=agency_resolution['mean'].values,

    y=agency_resolution.index

)



for container in ax.containers:

    ax.bar_label(container, fmt='%.2f', padding=4)



plt.title("Average Resolution Time by Agency")

plt.xlabel("Average Resolution Time (Hours)")

plt.ylabel("Agency")

plt.tight_layout()

plt.show()


agency_summary = df_clean.groupby('agency_name').agg(

    total_complaints=('unique_key', 'count'),

    closed_complaints=('status', lambda x: (x == 'Closed').sum())

)



agency_summary['closure_rate'] = (

    agency_summary['closed_complaints']

    / agency_summary['total_complaints'] * 100

)



print(

    agency_summary.sort_values(

        'total_complaints', ascending=False

    ).head(10).round(2)

)


# Convert created_date to datetime

df_clean['created_date'] = pd.to_datetime(

    df_clean['created_date'],

    errors='coerce'

)



# Extract date components

df_clean['year'] = df_clean['created_date'].dt.year

df_clean['month'] = df_clean['created_date'].dt.month

df_clean['month_name'] = df_clean['created_date'].dt.month_name()

df_clean['day_name'] = df_clean['created_date'].dt.day_name()

df_clean['hour'] = df_clean['created_date'].dt.hour

df_clean['date'] = df_clean['created_date'].dt.date



print(

    df_clean[

        ['created_date', 'year', 'month_name', 'day_name', 'hour']

    ].head()

)


daily_complaints = (

    df_clean.groupby('date')

    .size()

    .sort_index()

)



plt.figure(figsize=(14, 5))



daily_complaints.plot(kind='line', marker='o')



plt.title("Daily NYC 311 Complaint Trend")

plt.xlabel("Date")

plt.ylabel("Number of Complaints")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()


day_order = [

    'Monday', 'Tuesday', 'Wednesday',

    'Thursday', 'Friday', 'Saturday', 'Sunday'

]



day_counts = (

    df_clean['day_name']

    .value_counts()

    .reindex(day_order, fill_value=0)

)



plt.figure(figsize=(11, 5))



ax = sns.barplot(

    x=day_counts.index,

    y=day_counts.values

)



for container in ax.containers:

    ax.bar_label(container)



plt.title("NYC 311 Complaints by Day of Week")

plt.xlabel("Day")

plt.ylabel("Complaint Count")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


hourly_counts = (

    df_clean['hour']

    .value_counts()

    .reindex(range(24), fill_value=0)

)



plt.figure(figsize=(12, 5))



sns.lineplot(

    x=hourly_counts.index,

    y=hourly_counts.values,

    marker='o'

)



plt.title("NYC 311 Complaints by Hour")

plt.xlabel("Hour of Day (0â€“23)")

plt.ylabel("Complaint Count")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.show()


df_clean['year_month'] = (

    df_clean['created_date'].dt.to_period('M')

)



monthly_counts = (

    df_clean.groupby('year_month')

    .size()

    .sort_index()

)



plt.figure(figsize=(12, 5))



monthly_counts.plot(kind='bar')



plt.title("Monthly NYC 311 Complaints")

plt.xlabel("Year-Month")

plt.ylabel("Complaint Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


print("Highest complaint day:")

print(daily_complaints.idxmax(), daily_complaints.max())



print("\nHighest complaint weekday:")

print(day_counts.idxmax(), day_counts.max())



print("\nHighest complaint hour:")

print(hourly_counts.idxmax(), hourly_counts.max())


df_clean.to_csv(

    "urbanpulse_nyc_analyzed.csv",

    index=False

)







