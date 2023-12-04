import pandas as pd

f = r'C:\Users\S\Downloads\2019_kbo_for_kaggle_v2.csv'
df = pd.read_csv(f)

c22 = df.iloc[:, 21].apply(pd.to_numeric, errors='coerce')

filtered_df = df[(c22 >= 2015) & (c22 <= 2018)]

c7 = filtered_df.iloc[:, 6].apply(pd.to_numeric, errors='coerce')
top_10_values_c7 = c7.sort_values(ascending=False).head(10)

c10 = filtered_df.iloc[:, 9].apply(pd.to_numeric, errors='coerce')
c31 = filtered_df.iloc[:, 30].apply(pd.to_numeric, errors='coerce')
c32 = filtered_df.iloc[:, 31].apply(pd.to_numeric, errors='coerce')

max_row_10 = filtered_df.loc[c10.idxmax()]
max_row_31 = filtered_df.loc[c31.idxmax()]
max_row_32 = filtered_df.loc[c32.idxmax()]

print(top_10_values_c7)
print(max_row_10.iloc[0])
print(max_row_31.iloc[0])
print(max_row_32.iloc[0])





c24 = df.iloc[:, 23]
c27 = df.iloc[:, 26]

u = c27.unique()

m = {}
for i in u:
    msk = c27 == i
    r = df[msk].sort_values(by=c24.name, ascending=False).head(1)
    m[i] = r
for i, r in m.items():
    row = r.iloc[0]
    print(f"{i}: {row.iloc[0]}")







s = [6, 7, 10, 12, 13, 24, 31, 32, 33]
c = df.iloc[:, s].corrwith(df.iloc[:, 22])

m = c.idxmax(), c[c.idxmax()]
m
