
attribute_ID_val = 0
missing_val = 0
mean_val = 0.0
median_val = 0
sdev_val = 0.0
min_val = 0
max_val = 0
arity_val = 0

print("\nsymbolic attributes")
print("----------------------")
for i in range(0, num_columns):
    #is_string_dtype
    if pd.api.types.is_string_dtype(df[i]):
        print ("test")

x = df.house_value
print(x)
c = 'Charles_river_bound'
if pd.api.types.is_string_dtype(df[c]):
    print("true")

attribute = df.iloc[:,0]
print(attribute_names[1])
print(attribute)
print (scipy.stats.describe(attribute))
    #print(repr(x).rjust(8), repr(x*x).rjust(4),repr(x*x*x).rjust(5))
