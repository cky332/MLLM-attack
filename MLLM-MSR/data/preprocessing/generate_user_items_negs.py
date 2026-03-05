"""
Generate user_items_negs.tsv from MicroLens-50k_pairs.csv

This script performs:
1. Load raw interaction data (user, item, timestamp)
2. K-core filtering (min 7 interactions per user, min 5 interactions per item)
3. Select top-6 most frequent items per user (sorted by timestamp)
4. Random negative sampling (20 negative items per user)
5. Output user_items_negs.tsv

Usage:
    cd ~/MLLM/MLLM-MSR/data/preprocessing
    python generate_user_items_negs.py
"""

import os
import pandas as pd
import numpy as np
from collections import Counter

# ============ Step 1: Load data ============
csv_path = os.path.join(os.path.dirname(__file__), '..', 'microlens', 'MicroLens-50k_pairs.csv')
csv_path = os.path.abspath(csv_path)
print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Original shape: {df.shape}")
print(df.head())

# ============ Step 2: K-core filtering ============
min_u_num, min_i_num = 7, 5


def get_illegal_ids_by_inter_num(df, field, max_num=None, min_num=None):
    if field is None:
        return set()
    if max_num is None and min_num is None:
        return set()

    max_num = max_num or np.inf
    min_num = min_num or -1

    ids = df[field].values
    inter_num = Counter(ids)
    ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
    print(f'{len(ids)} illegal_ids_by_inter_num, field={field}')
    return ids


def filter_by_k_core(df):
    iteration = 0
    while True:
        iteration += 1
        ban_users = get_illegal_ids_by_inter_num(df, field='user', max_num=None, min_num=min_u_num)
        ban_items = get_illegal_ids_by_inter_num(df, field='item', max_num=None, min_num=min_i_num)
        if len(ban_users) == 0 and len(ban_items) == 0:
            print(f"K-core filtering converged after {iteration} iterations")
            return

        dropped_inter = pd.Series(False, index=df.index)
        if 'user':
            dropped_inter |= df['user'].isin(ban_users)
        if 'item':
            dropped_inter |= df['item'].isin(ban_items)
        print(f'{dropped_inter.sum()} dropped interactions (iteration {iteration})')
        df.drop(df.index[dropped_inter], inplace=True)


filter_by_k_core(df)
print(f"Shape after k-core filtering: {df.shape}")
print(f"Unique users: {df['user'].nunique()}")
print(f"Unique items: {df['item'].nunique()}")

# ============ Step 3: Calculate item frequency and sort by timestamp ============
item_frequency = df.groupby('item').size().reset_index(name='frequency')
item_frequency_sorted = item_frequency.sort_values(by='frequency', ascending=False)
df_sorted = df.sort_values('timestamp')

all_items = set(df_sorted['item'].unique())
all_users = df_sorted['user'].unique()

# ============ Step 4: Generate negative samples ============
print("Generating negative samples for each user...")
negative_samples_per_user = {}
for i, user in enumerate(all_users):
    user_items = df_sorted[df_sorted['user'] == user]['item'].unique()
    available_items = list(all_items - set(user_items))
    negative_samples = np.random.choice(available_items, size=min(20, len(available_items)), replace=False)
    negative_samples_per_user[user] = negative_samples
    if (i + 1) % 5000 == 0:
        print(f"  Processed {i + 1}/{len(all_users)} users")

print(f"Generated negative samples for {len(negative_samples_per_user)} users")

# ============ Step 5: Select top-6 items per user ============
print("Selecting top-6 frequent items per user...")
item_freq_map = item_frequency_sorted.set_index('item')['frequency']
top_6_items_per_user = {}
for i, user in enumerate(all_users):
    user_df = df_sorted[df_sorted['user'] == user]
    user_items = user_df['item']
    # Get frequency for each item
    item_freqs = user_items.map(item_freq_map)
    # Get indices of top-6 by frequency
    top_indices = item_freqs.sort_values(ascending=False).index[:6]
    # Sort these top-6 items by timestamp
    top_6_items_per_user[user] = df_sorted.loc[top_indices].sort_values('timestamp')['item'].values
    if (i + 1) % 5000 == 0:
        print(f"  Processed {i + 1}/{len(all_users)} users")

print(f"Selected top-6 items for {len(top_6_items_per_user)} users")

# ============ Step 6: Write to user_items_negs.tsv ============
output_dir = os.path.join(os.path.dirname(__file__), '..', 'MicroLens-50k', 'Split')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'user_items_negs.tsv')

lines = []
for user in all_users:
    top_items_str = ', '.join([str(item) for item in top_6_items_per_user[user]])
    negative_samples_str = ', '.join([str(item) for item in negative_samples_per_user[user]])
    line = f"{user}\t{top_items_str}\t{negative_samples_str}"
    lines.append(line)

with open(output_path, 'w') as file:
    file.write('\n'.join(lines))

print(f"\nDone! File saved to: {output_path}")
print(f"Total users: {len(lines)}")
print(f"Sample line: {lines[0]}")

# Also copy titles to the expected location
titles_src = os.path.join(os.path.dirname(__file__), '..', 'microlens', 'MicroLens-50k_titles.csv')
titles_dst = os.path.join(os.path.dirname(__file__), '..', 'MicroLens-50k', 'MicroLens-50k_titles.csv')
if os.path.exists(titles_src) and not os.path.exists(titles_dst):
    import shutil
    shutil.copy2(titles_src, titles_dst)
    print(f"Copied titles file to: {titles_dst}")
