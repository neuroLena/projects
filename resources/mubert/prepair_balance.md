```
n_down=sum(clean_df['downloaded']==1) #76798
n_notdown= sum(clean_df['downloaded']==0) #503054
print(f'Downloaded tracks: {n_down}')
print(f'Not downloaded tracks: {n_notdown}')

import random
random.seed(42)

# Down-sampling
df0 = clean_df[clean_df['downloaded']==0].sample(len(clean_df[clean_df['downloaded']==1]))
balanced_df = pd.concat([clean_df[clean_df['downloaded']==1], df0], ignore_index=True)

# Up-sampling
# df1 = clean_df[clean_df['downloaded']==1].sample(len(clean_df[clean_df['downloaded']==0]), replace=True)
# df_balanced = pd.concat([clean_df[clean_df['downloaded']==0], df1], ignore_index=True)

# Sample specified number
# n_samp=5000
# balanced_df = pd.concat([clean_df[clean_df['downloaded']==0].sample(n_samp, replace=True), 
#                         clean_df[clean_df['downloaded']==1].sample(n_samp, replace=True)], 
#                         ignore_index=True)

balanced_df = balanced_df.drop('Unnamed: 0', axis=1)
df=balanced_df
print(f'Balanced data size: {df.shape}')
df.head()
```