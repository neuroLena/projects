
```
data = pd.read_csv(f'{dir_path}/data/ttm_dump.csv')
tags = pd.read_json(f'{dir_path}/data/sid_tags.json', orient='index')


# Rename columns
data = data.rename(columns={'stream_id': 'sid'})
data = data.rename(columns={'downloaded_count': 'downloaded'})
data = data.rename(columns={'name': 'input'})

# Sids to column
tags['sid']=tags.index

# Tags columns to list
tags = pd.DataFrame(tags)
tags['tags'] = tags.apply(lambda row: [val for val in row.values if val is not None], axis=1)


# Check if any sid has no tags = no such sid
n_notags=0
for i in range(len(tags.index)):
    if not tags.iloc[100].isna().any():
        n_notags+=1
    else:
        pass
print(f'Sids with no tags: {n_notags}')
    
# Check missing sids 
n_nosid=sum(data_df['sid'].isna())
# tags['sid'].unique() #= no missing sids
print(f'Rows in @data with no sids: {n_nosid}')

# Check if there are sids in @tags that are not present in @data 
set1=set(data_df['sid'][data_df['sid'].notnull()].unique().astype(int))
set2=set(tags_df['sid'])
diff12 = set1.difference(set2) #= [76488, 81786, 81923, 81938, 81985, 82038, 82099]
diff12
print(f'Sids in @data not present in @tags: {list(diff12)}')

# Check if there are sids in 'tags' hat are not present in 'data'
diff21=set2.difference(set1) #= all from 'data' are in 'tags'
print(f'Sids in @tags not present in @data:{list(diff21)}')


# Exclude nonoverlaping sids
data_df = data_df[~data_df['sid'].isin(diff12)]

# Subset data with sids
nosid_df = data[data['sid'].isna()]
sid_df = data[data['sid'].notnull()]

# Merge  datasets by 'sid' column
merged_df = pd.merge(sid_df, tags_df, on='sid', how='left')

# Cleaned version
clean_df = merged_df
clean_df = clean_df.rename(columns={'sid_x': 'sid'})

# Here I exclude options with more then 1 loads [for now] 
more1=sum(merged_df['downloaded']>1)
print(f'Rows with more then 1 downloads: {more1}')
clean_df = clean_df[merged_df['downloaded'].isin([0, 1])]
clean_df.iloc[:,[7,4]]= clean_df.iloc[:,[7,4]].astype(int)
```

