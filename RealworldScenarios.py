import spark

from pyspark import SparkContext
from pyspark.sql import SparkSession

# Initialize a Spark session and context
spark = SparkSession.builder.appName("example").getOrCreate()
sc = spark.sparkContext  # Access the SparkContext from the Spark session


# 1. Parsing and Flattening JSON Data
# Many times, data comes in a nested JSON format, and we need to flatten it to make it easier to work with. Let's parse a JSON file and flatten it.

# Scenario: Parsing JSON Data

# Sample JSON data
data = [
    '{"id": 1, "name": "John", "address": {"city": "New York", "zip": "10001"}}',
    '{"id": 2, "name": "Jane", "address": {"city": "San Francisco", "zip": "94105"}}'
]

# Create a DataFrame from the JSON data
df = spark.read.json(sc.parallelize(data))

# Show the original JSON data
df.show(truncate=False)


spark.read.json()


# Flatten the nested "address" column to separate columns for "city" and "zip"
flattened_df = df.select(
    "id",
    "name",
    "address.city",
    "address.zip"
)

# Show the flattened DataFrame
flattened_df.show(truncate=False)

df.select("address.city", "address.zip") #flattens the nested address column into individual fields, making it easier to process.


# -------------------------------------------
# 2. Working with Complex Nested Structures
# Scenario: Handling Complex JSON with Arrays
# Consider a more complex scenario where you have JSON data with arrays. Let’s say you have an array of items purchased in each order, and you need to analyze them.

# Sample JSON data with an array of items
data = [
    '{"order_id": 1, "customer": "John", "items": [{"item": "apple", "quantity": 2}, {"item": "banana", "quantity": 3}]}',
    '{"order_id": 2, "customer": "Jane", "items": [{"item": "orange", "quantity": 1}]}'
]

# Create DataFrame from JSON
df = spark.read.json(sc.parallelize(data))

# Show the DataFrame with nested arrays
df.show(truncate=False)

# The items field is an array, and each item is a struct containing the item name and quantity.

from pyspark.sql.functions import explode

# Use explode to flatten the array of items
flattened_df = df.withColumn("item", explode(df.items)) \
    .select("order_id", "customer", "item.item", "item.quantity")

# Show the flattened DataFrame
flattened_df.show(truncate=False)


# explode(df.items) flattens the array of items, turning each item into a new row while maintaining the original order_id and customer values.

# The item.item and item.quantity allow access to the fields within each item struct.


# --------------------------------------------------------------
# 3. Joining DataFrames with Different Data Sources (ETL Process)
# Often, data comes from multiple sources, and we need to join them. For instance, you may want to join a sales record with a product catalog.

# Scenario: Joining DataFrames from Different Sources

# Sample DataFrames
sales_data = [
    (1, "apple", 100),
    (2, "banana", 150),
    (3, "orange", 120)
]

catalog_data = [
    ("apple", "fruit"),
    ("banana", "fruit"),
    ("orange", "fruit"),
    ("carrot", "vegetable")
]

# Create DataFrames
sales_df = spark.createDataFrame(sales_data, ["id", "product", "amount"])
catalog_df = spark.createDataFrame(catalog_data, ["product", "category"])

# Perform an inner join on "product"
joined_df = sales_df.join(catalog_df, "product")

# Show the joined DataFrame
joined_df.show()


# We created two DataFrames, sales_df (sales records) and catalog_df (product catalog).

# We used .join() to join them on the common column product.

# Scenario: Left Join with a Default Value for Missing Data

# Perform a left join to include all sales records, even if the product is missing in the catalog
left_joined_df = sales_df.join(catalog_df, "product", "left_outer")

# Replace null values in the category column with a default value
left_joined_df = left_joined_df.fillna({"category": "unknown"})

# Show the result
left_joined_df.show()
# Explanation:

# .join("product", "left_outer") ensures that all rows from sales_df are kept, even if there's no match in catalog_df.

# fillna({"category": "unknown"}) fills null values in the category column with "unknown".


# --------------------------------------------------
# 4. Aggregating Data and Calculating Metrics
# Data is often aggregated to provide useful insights. For instance, summarizing sales data or calculating averages.

# Scenario: Aggregating Sales Data by Product

# Sample sales data with product and amount sold
sales_data = [
    ("apple", 10),
    ("banana", 20),
    ("apple", 15),
    ("banana", 25),
    ("orange", 30)
]

# Create DataFrame
df = spark.createDataFrame(sales_data, ["product", "amount"])

# Group by product and calculate total sales
aggregated_df = df.groupBy("product").sum("amount").withColumnRenamed("sum(amount)", "total_sales")

# Show the aggregated data
aggregated_df.show()
# Explanation:

# .groupBy("product").sum("amount") groups data by the product column and calculates the sum of the amount column for each product.

# .withColumnRenamed("sum(amount)", "total_sales") renames the aggregation column to total_sales.

# Scenario: Calculating Average Sales Per Product

# Calculate the average sales per product
avg_sales_df = df.groupBy("product").avg("amount").withColumnRenamed("avg(amount)", "average_sales")

# Show the result
avg_sales_df.show()
# Explanation:

# .groupBy("product").avg("amount") calculates the average sales amount for each product.



# ---------------------------------------------------
# 5. Time Series Data Processing
# Time series data is another real-world scenario where you may need to perform operations like resampling, aggregations, or smoothing.

# Scenario: Resampling Time Series Data

# Sample time series data
from pyspark.sql.functions import to_timestamp

data = [
    ("2023-01-01", 100),
    ("2023-01-02", 150),
    ("2023-01-03", 120)
]

# Create DataFrame
df = spark.createDataFrame(data, ["date", "sales"])

# Convert the "date" column to timestamp format
df = df.withColumn("date", to_timestamp("date", "yyyy-MM-dd"))

# Show the DataFrame with date as timestamp
df.show()
# Explanation:

# The to_timestamp function converts a string to a timestamp format.

# This is important when working with time series data.

# Scenario: Rolling Window Calculations (Moving Average)
from pyspark.sql.window import Window
from pyspark.sql.functions import avg

# Define a window specification
windowSpec = Window.orderBy("date").rowsBetween(-2, 0)

# Calculate the moving average over a 3-day window
df_with_avg = df.withColumn("moving_avg", avg("sales").over(windowSpec))

# Show the result
df_with_avg.show()
# Explanation:

# Window.orderBy("date").rowsBetween(-2, 0)
# # defines a window to calculate the moving average over a 3-day rolling window.

# avg("sales").over(windowSpec) 
# # calculates the moving average over the defined window.

# --------------------------------------------------------------

# 6. Handling Large Datasets: Partitioning and Caching
# In real-world scenarios, large datasets require optimization techniques like partitioning and caching to improve performance.

# Scenario: Repartitioning Data

# Repartition a DataFrame into 4 partitions for parallel processing
df_repartitioned = df.repartition(4)

# Show the number of partitions
print(df_repartitioned.rdd.getNumPartitions())  # Output: 4
# Explanation:

# .repartition(4) increases the number of partitions, allowing for better parallelism when performing transformations.

# Scenario: Caching Data for Performance

# Cache the DataFrame to improve performance for repeated operations
df.cache()

# Perform transformations on the cached DataFrame
df_filtered = df.filter(df.sales > 120)
df_filtered.show()
# Explanation:

# .cache() stores the DataFrame in memory, which speeds up repeated operations.