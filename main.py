from helpers import data_reading_and_understanding as dr

# Task 1: Data understanding and preapearing

# Step 1: Data reading

dr.pd.set_option("display.float_format", lambda x: "%.3f" % x)
dr.pd.set_option('display.width', 99)
dr.pd.set_option("display.max_columns", None)


df = dr.dataframe_reading("datasets/x_shoe_company.csv")

# Step 2: Understanding data

dr.check_data(df)

# Step 3: Creating total shopping and total shopping value variables

df["omni_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df["omni_total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]


# Step 4: Changing to date some columns that is expressing to date

for col in df.columns:
    if "date" in col:
        df[col] = dr.pd.to_datetime(df[col])


# Step 5.1: Check the number of customer in the shopping channels and total number of product

df.groupby("order_channel").agg({"master_id": lambda master_id: master_id.nunique(),
                                 "order_num_total_ever_online": lambda order_num_total_ever_online: order_num_total_ever_online.sum(),
                                 "customer_value_total_ever_online": lambda customer_value_total_ever_online: customer_value_total_ever_online.sum()})

df["order_channel"].unique()
df["last_order_channel"].unique()
df.groupby("order_channel").agg({"master_id":"count",
                                 "omni_total_order":"sum",
                                 "omni_customer_value":"sum"})

# Step 5.2: Check the number of customer in the shopping channels and total number of product

df.groupby("order_channel").agg({"master_id": lambda master_id: master_id.nunique(),
                                 "omni_total_order": lambda omni_total_order: omni_total_order.sum(),
                                 "omni_customer_value": lambda omni_customer_value: omni_customer_value.sum()})

# Step 6: Top ten customers that are spent maximum money

df.sort_values(by='omni_customer_value', ascending=False).head(10)

# Step 7: Top ten customers that are maximum order

df_top_ten_order = df.sort_values(by="omni_total_order", ascending=False).head(10)


# Step 8: Do it a function all data preparation

def data_preparation(dataframe):
    dataframe["omni_customer_value"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    dataframe["omni_total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["first_order_date"] = dr.pd.to_datetime(dataframe["first_order_date"])
    dataframe["last_order_date"] = dr.pd.to_datetime(dataframe["last_order_date"])
    dataframe["last_order_date_online"] = dr.pd.to_datetime(dataframe["last_order_date_online"])
    dataframe["last_order_date_offline"] = dr.pd.to_datetime(dataframe["last_order_date_offline"])
    return dataframe

data_preparation(df)

# Task 2: Calculating RFM Metrics

# Step 1: Do definition rfm metrics

# Recency: It representing difference the between customers first order date and analysis date.
# Frequency: It representing customers number of total order.
# Monetary: It representing the customers total money value bring to company

# Step 2: Calculate the RFM Metrics specifically customer

today_date = dr.dt.datetime(2021, 6, 1)
df["last_order_date"].max()
rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                             "omni_total_order": lambda omni_total_order: omni_total_order,
                             "omni_customer_value": lambda omni_customer_value: omni_customer_value})
df["last_order_date"].min()

date_var = [col for col in df.columns if 'date' in col]

for col in date_var:
    df[col] = dr.pd.to_datetime(df[col])

df[date_var].max().max()

# Step 3: Change the metrics names that are you created in step 2. As recency, frequency and monetary

rfm.columns = ["recency", "frequency", "monetary"]

# you can also do it like
rfm.rename(columns={"last_order_date":"recency","omni_total_order":"frequency","omni_customer_value":"monetary"})

# Task 3: Calculate the RF metrics

rfm["recency_score"] = dr.pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = dr.pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["RF_score"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
rfm.reset_index(inplace=True)

# Task 4: Describe, RF metrics as segmentation

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["Segment"] = rfm["RF_score"].replace(seg_map, regex=True)

# Step 5: Action time

# Task 1: Observe the segmentations average

rfm[["Segment", "recency", "frequency", "monetary"]].groupby("Segment").agg(["mean", "count"])

# Task 2: With the help of RFM analysis, find the customers in the relevant profile for the 2 cases given below and save the customer IDs as csv.

# X shoe company is introducing a new women's shoe brand within its portfolio.
# The prices of the products from this new brand are higher than the general customer preferences.
# Therefore, there is a desire to reach out to customers who are likely to be interested in the promotion and sales of these products.
# Specifically, communication will be established with loyal customers (champions, loyal_customers) and individuals who have shown interest in the women's category within the shopping category.
# The IDs of these customers will be saved to a CSV file.

rfm1 = rfm[rfm["Segment"].str.contains("champions|loyal_customers")][["master_id", "Segment"]]

df1 = df[df["interested_in_categories_12"].str.contains("KADIN", na=False)][["master_id", "interested_in_categories_12"]]

cus_for_new_brand = rfm1.merge(df1, on="master_id", how="inner")

cus_for_new_brand.to_csv("cus_for_new_brand")

# A discount of nearly 40% is planned for men's and children's products.
# Good customers who have shown interest in the relevant categories in the past but have not made purchases for a long time, dormant customers, and new customers are specifically targeted to ensure they are not lost.
# The IDs of customers with suitable profiles will be saved to a CSV file.

rfm2 = rfm[rfm["Segment"].str.contains("cant_loose|hibernating|new_customers")][["master_id", "Segment"]]
df2 = df[(df["interested_in_categories_12"].str.contains("ERKEK|COCUK", na=False))][["master_id", "interested_in_categories_12"]]
cus_for_new_brand2 = rfm2.merge(df2, on="master_id", how="inner")
cus_for_new_brand2.to_csv("cus_for_new_brand2")





