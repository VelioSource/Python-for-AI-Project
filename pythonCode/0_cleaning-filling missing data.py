import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Project\Large set\large_dataset.csv")
fs = pd.read_csv(r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Project\Fighter stats\fighter_stats.csv")
output_file = r"C:\Users\Velimir\iCloudDrive\Documents\Python for AI\Project\Large set\ufc_processed.csv"


#Dropping columns 

cols_to_drop = [
    #Useless, model should predict outcome based on skill, not names
    "event_name", "r_fighter", "b_fighter", "referee",

    #Reach - too many missing values
    "r_reach", "b_reach", "reach_diff",

    #Fight Outocomes - Data Leakage
    "method", "finish_round", "time_sec",

    #Direct Fights data, pure data leakage
    "r_kd","r_sig_str","r_sig_str_att","r_sig_str_acc",
    "r_str","r_str_att","r_str_acc","r_td","r_td_att","r_td_acc",
    "r_sub_att","r_rev","r_ctrl_sec",
    
    "b_kd","b_sig_str","b_sig_str_att","b_sig_str_acc",
    "b_str","b_str_att","b_str_acc","b_td","b_td_att","b_td_acc",
    "b_sub_att","b_rev","b_ctrl_sec",

    #Difference between fighters DURING the match- Data leakage
    "kd_diff", "sig_str_diff", "sig_str_att_diff","sig_str_acc_diff",
    "str_diff", "str_att_diff","str_acc_diff","td_diff","td_att_diff",
    "td_acc_diff", "sub_att_diff","rev_diff", "ctrl_sec_diff"
]

df = df.drop(columns=cols_to_drop, errors="ignore")

print(df.shape, df.columns) 

#df.to_csv(output_file, index=False)
df.isnull().sum().sort_values(ascending=False).head(10)


#Filling missing age values - Median
for col in ["r_age", "b_age"]:
    df[col] = df[col].fillna(df[col].median())


df["age_diff"] = df["r_age"] - df["b_age"]

#Filling missing stance - KNN, using height, weight, wins losses as neighbours
#Look at 5 fighters with similar features and choose the most suiting stance

for prefix in ["r","b"]:
    stance_col = f'{prefix}_stance'
    features = [f"{prefix}_height",f"{prefix}_weight",f"{prefix}_wins_total", f"{prefix}_losses_total"]

    known_stance = df[stance_col].notnull()
    unknown_stance = df[stance_col].isnull()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(df.loc[known_stance,features], df.loc[known_stance, stance_col])
    df.loc[unknown_stance, stance_col] = knn.predict(df.loc[unknown_stance, features])


#Categorical to numerical data 
#If red == 1
df["winner"] = (df["winner"] == "Red").astype(int)

#if Man == 1
df["gender"] = (df["gender"] == "Men").astype(int)

weight_order = [
    "Women's Strawweight",    # 0  (115 lb)
    "Women's Flyweight",      # 1  (125 lb)
    "Women's Bantamweight",   # 2  (135 lb)
    "Women's Featherweight",  # 3  (145 lb)
    "Flyweight",              # 4  (125 lb men)
    "Bantamweight",           # 5  (135 lb)
    "Featherweight",          # 6  (145 lb)
    "Lightweight",            # 7  (155 lb)
    "Welterweight",           # 8  (170 lb)
    "Middleweight",           # 9  (185 lb)
    "Light Heavyweight",      # 10 (205 lb)
    "Heavyweight",            # 11 (265 lb)
    "Super Heavyweight",      # 12
    "Catch Weight",           # 13
    "Open Weight",            # 14
]
#Transform to number:categorical_devision
#wc_map["Featherweight"] -> 6
wc_map = {w: i for i, w in enumerate(weight_order)}

def encode_weight_class(val):
    if pd.isnull(val):
        return -1
    # exact match first
    if val in wc_map:
        return wc_map[val]
    # substring match for values like "UFC Featherweight Title"
    for weight_name, code in wc_map.items():
        if weight_name in val:
            return code
    return -1

df["weight_class"] = df["weight_class"].apply(encode_weight_class)

#Stance categorical to numerical

stance_map = {
    'Orthodox':    0,
    'Southpaw':    1,
    'Switch':      2,
    'Open Stance': 3,
    'Sideways':    4,
}

df["r_stance"] = df["r_stance"].map(stance_map).fillna(-1).astype(int)
df["b_stance"] = df["b_stance"].map(stance_map).fillna(-1).astype(int)

#Total rounds - FIlling missing values - if title = 5 rounds, else 3 rounds

missing_rounds = df["total_rounds"].isnull()

title_fights = missing_rounds & (df["is_title_bout"] == 1)
non_title_fights = missing_rounds & (df["is_title_bout"] == 0)

df.loc[title_fights, "total_rounds"] = 5
df.loc[non_title_fights, "total_rounds"] = 3

#Final summary

print("\n=== FINAL DATASET ===")
print(f"Shape:             {df.shape}")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print(f"All numeric:       {all(df.dtypes != object)}")

print(f"\nEncoding reference:")
print(f"  winner       -> Red=1, Blue=0  (target)")
print(f"  gender       -> Men=1, Women=0")
print(f"  weight_class -> ordinal 0-14 by weight (Women's Strawweight=0 ... Heavyweight=11)")
print(f"  r/b_stance   -> Orthodox=0, Southpaw=1, Switch=2, Open Stance=3, Sideways=4")
 


df.to_csv(output_file, index=False)
print("\nSaved -> ufc_processed.csv")


    
