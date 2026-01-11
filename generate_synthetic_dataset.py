import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import math
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT_PATH = "./backend"
N_ROWS = 100_000
N_CUSTOMERS = 5000
N_MERCHANTS = 500
FRAUD_RATE = 0.02  # 2%

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 3, 31)  # inclusive-ish

# Merchant categories (as in PDF)
MERCHANT_CATEGORIES = ["grocery", "electronics", "gas", "restaurant", "retail", "jewelry", "luxury_goods"]

# Indian bounding box
LAT_MIN, LAT_MAX = 8.4, 37.6
LON_MIN, LON_MAX = 68.7, 97.25

# Urban center seeds (lat, lon, weight) for realistic homes/merchants
URBAN_CENTERS = [
    (28.7041, 77.1025, 1.0),  # Delhi
    (19.0760, 72.8777, 1.0),  # Mumbai
    (12.9716, 77.5946, 0.8),  # Bangalore
    (13.0827, 80.2707, 0.6),  # Chennai
    (22.5726, 88.3639, 0.5),  # Kolkata
    (26.9124, 75.7873, 0.3),  # Jaipur
]

# helper: Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# 1) Create customers with home centroids and profiles
customers = []
center_weights = np.array([c[2] for c in URBAN_CENTERS])
center_weights = center_weights / center_weights.sum()
for i in range(1, N_CUSTOMERS + 1):
    cid = f"CUST_{i:05d}"
    # pick urban center
    idx = np.random.choice(len(URBAN_CENTERS), p=center_weights)
    base_lat, base_lon = URBAN_CENTERS[idx][0], URBAN_CENTERS[idx][1]
    # jitter home location: most within 0-20 km for city
    home_lat = base_lat + np.random.normal(scale=0.08)  # ~ few km jitter
    home_lon = base_lon + np.random.normal(scale=0.08)
    # spending profile: conservative / normal / high
    profile = np.random.choice(["conservative", "normal", "high"], p=[0.5, 0.4, 0.1])
    base_rate = {"conservative": 0.6, "normal": 1.0, "high": 2.0}[profile]
    # category preferences: assign probabilities biased to a few categories
    cat_probs = np.random.dirichlet(np.ones(len(MERCHANT_CATEGORIES)))
    customers.append((cid, home_lat, home_lon, profile, base_rate, cat_probs.tolist()))

cust_df = pd.DataFrame(customers, columns=["customer_id", "home_lat", "home_long", "profile", "base_rate", "cat_probs"])

# 2) Create merchants: category, location, and an "is_collusive" flag candidate
merchants = []
# weight urban centers for merchants as well
merchant_center_weights = center_weights
for i in range(1, N_MERCHANTS + 1):
    mid = f"MERCHANT_{i:04d}"
    category = np.random.choice(MERCHANT_CATEGORIES, p=[0.25, 0.12, 0.12, 0.18, 0.18, 0.08, 0.07])
    idx = np.random.choice(len(URBAN_CENTERS), p=merchant_center_weights)
    base_lat, base_lon = URBAN_CENTERS[idx][0], URBAN_CENTERS[idx][1]
    # merchant location jitter depends on category (grocery close, luxury can be spread)
    sigma = {"grocery": 0.02, "gas": 0.02, "restaurant": 0.03, "retail": 0.04, "electronics": 0.06, "jewelry": 0.06, "luxury_goods":0.08}
    mlat = base_lat + np.random.normal(scale=sigma[category])
    mlon = base_lon + np.random.normal(scale=sigma[category])
    merchants.append((mid, category, mlat, mlon))
merch_df = pd.DataFrame(merchants, columns=["merchant_id", "merchant_category", "merchant_lat", "merchant_long"])

# choose collusive merchants (for merchant_collusion)
NUM_COLLUSIVE_MERCHANTS = 10
collusive_merchants = merch_df.sample(n=NUM_COLLUSIVE_MERCHANTS, random_state=SEED).merchant_id.tolist()

# 3) Generate N_ROWS raw base transactions by sampling customers & merchants (legit baseline)
# compute per-customer selection probabilities from base_rate
cust_probs = cust_df["base_rate"].values
cust_probs = cust_probs / cust_probs.sum()

# sample customers for each transaction (vectorized)
sampled_customer_indices = np.random.choice(np.arange(N_CUSTOMERS), size=N_ROWS, p=cust_probs)
sampled_customers = cust_df.iloc[sampled_customer_indices].reset_index(drop=True)

# For each sampled customer, select a merchant based on their category preference
merchant_by_category = {cat: merch_df[merch_df["merchant_category"] == cat]["merchant_id"].tolist() for cat in MERCHANT_CATEGORIES}

txn_rows = []
# pre-generate timestamp days uniformly and then sample hour by seasonality
total_seconds = int((END_DATE - START_DATE).total_seconds())
# Hour distribution: bimodal - lunch and evening, low night activity
hours = np.arange(24)
hour_probs = np.zeros(24)
# lunch 12-14 smaller peak
hour_probs[12:15] += 0.18 / 3
# evening 18-21 larger peak
hour_probs[18:22] += 0.40 / 4
# morning 8-10 small
hour_probs[8:11] += 0.12 / 3
# rest distributed
remaining = 1.0 - hour_probs.sum()
hour_probs += remaining / 24
hour_probs = hour_probs / hour_probs.sum()

# Day of week probs: weekend slightly higher for restaurants
dow_probs = np.array([0.12,0.14,0.14,0.14,0.14,0.16,0.16])
dow_probs = dow_probs / dow_probs.sum()

# Amount log-normal params per category (mu, sigma on log scale)
amt_params = {
    "grocery": (np.log(500), 0.6),
    "gas": (np.log(2000), 0.4),
    "restaurant": (np.log(1200), 0.7),
    "retail": (np.log(2500), 0.9),
    "electronics": (np.log(15000), 1.1),
    "jewelry": (np.log(35000), 1.2),
    "luxury_goods": (np.log(40000), 1.3),
}

# Prepare card_number mapping: 1 primary card per customer (simple)
card_numbers = [f"CARD_{100000 + i}" for i in range(1, N_CUSTOMERS+1)]
cust_card_map = dict(zip(cust_df.customer_id, card_numbers))

# Create transactions by vectorized loops (fast enough)
for idx in range(N_ROWS):
    cust_row = sampled_customers.iloc[idx]
    cid = cust_row["customer_id"]
    # choose merchant category by customer's category probs
    cat_probs = np.array(cust_row["cat_probs"])
    cat_probs = cat_probs / cat_probs.sum()
    mcat = np.random.choice(MERCHANT_CATEGORIES, p=cat_probs)
    # pick a merchant id from that category uniformly
    mlist = merchant_by_category[mcat]
    if len(mlist) == 0:
        merch_choice = merch_df.sample(n=1).merchant_id.values[0]
    else:
        merch_choice = random.choice(mlist)
    merch_info = merch_df[merch_df["merchant_id"] == merch_choice].iloc[0]
    mlat = merch_info["merchant_lat"]
    mlon = merch_info["merchant_long"]
    # timestamp: choose a random second in period then alter hour using hour_probs and dow_probs
    rand_seconds = random.randint(0, total_seconds)
    ts = START_DATE + timedelta(seconds=rand_seconds)
    # resample hour from hour_probs and replace
    selected_hour = np.random.choice(hours, p=hour_probs)
    ts = ts.replace(hour=int(selected_hour), minute=int(np.random.randint(0,60)), second=int(np.random.randint(0,60)))
    # amount: log-normal per category
    mu, sigma = amt_params[mcat]
    amount = float(np.exp(np.random.normal(mu, sigma)))
    # round a bit: keep rupee cents not important, round to 2 decimals
    amount = round(max(1.0, amount), 2)
    # compute distance from home
    home_lat = cust_row["home_lat"]
    home_lon = cust_row["home_long"]
    distance = round(haversine(home_lat, home_lon, mlat, mlon), 2)
    txn_rows.append({
        "customer_id": cid,
        "card_number": cust_card_map[cid],
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "amount": amount,
        "merchant_id": merch_choice,
        "merchant_category": mcat,
        "merchant_lat": float(round(mlat,6)),
        "merchant_long": float(round(mlon,6)),
        "home_lat": float(round(home_lat,6)),
        "home_long": float(round(home_lon,6)),
        "distance_from_home": distance
    })

base_df = pd.DataFrame(txn_rows)

# Insert transaction_id and other columns defaults
base_df.insert(0, "transaction_id", [f"TXN_{i:08d}" for i in range(1, N_ROWS+1)])
base_df["is_fraud"] = 0
base_df["fraud_type"] = "none"
# hour, day_of_week, month columns derived
base_df["ts_dt"] = pd.to_datetime(base_df["timestamp"])
base_df["hour"] = base_df["ts_dt"].dt.hour
base_df["day_of_week"] = base_df["ts_dt"].dt.weekday
base_df["month"] = base_df["ts_dt"].dt.month

# 4) Inject fraud patterns to reach FRAUD_RATE (approx)
target_frauds = int(N_ROWS * FRAUD_RATE)
remaining_to_inject = target_frauds

# proportions among types (tunable)
pct_cloning = 0.4
pct_ato = 0.4
pct_collusion = 0.2

n_cloning = int(target_frauds * pct_cloning)
n_ato = int(target_frauds * pct_ato)
n_collusion = target_frauds - n_cloning - n_ato

# Helper indexes
df = base_df  # rename

# --- Inject Merchant Collusion: pick collusive merchants and compromised customers (mules)
mule_customers = df["customer_id"].drop_duplicates().sample(n=50, random_state=SEED).tolist()
coll_merchants = collusive_merchants
# Pick transactions belonging to those mules and set a lot to collusive merchants
coll_txn_indices = df[df["customer_id"].isin(mule_customers)].sample(n=min(700, len(df)), random_state=SEED).index.tolist()
# Force those transactions to be at collusive merchants, structured amounts
struct_amounts = [49000, 49500, 49999, 25000, 5000, 10000]
for i, idx in enumerate(coll_txn_indices):
    m = random.choice(coll_merchants)
    df.at[idx, "merchant_id"] = m
    # set merchant_category to a collusive category (jewelry/electronics/consulting)
    coll_cat = random.choice(["jewelry", "electronics", "luxury_goods"])
    df.at[idx, "merchant_category"] = coll_cat
    # set merchant coords from merchant table
    merch_row = merch_df[merch_df["merchant_id"] == m].iloc[0]
    df.at[idx, "merchant_lat"] = merch_row["merchant_lat"]
    df.at[idx, "merchant_long"] = merch_row["merchant_long"]
    # amount structuring near PAN threshold or round numbers
    if random.random() < 0.6:
        amt = float(random.choice([49000, 49500, 49999]))
    else:
        amt = float(random.choice([5000,10000,25000]))
    df.at[idx, "amount"] = amt
    # recompute distance
    dist = haversine(df.at[idx,"home_lat"], df.at[idx,"home_long"], df.at[idx,"merchant_lat"], df.at[idx,"merchant_long"])
    df.at[idx,"distance_from_home"] = round(dist,2)
    # mark some as fraud
for idx in coll_txn_indices[:n_collusion]:
    df.at[idx, "is_fraud"] = 1
    df.at[idx, "fraud_type"] = "merchant_collusion"
remaining_to_inject -= min(n_collusion, len(coll_txn_indices))

# --- Inject Card Cloning: impossible travel & concurrent sessions
# For cloning, we create events: take an existing txn t1, and create/modify a t2 for same card at far location with tiny dt
cloning_candidates = df.sample(n=2000, random_state=SEED+1).index.tolist()  # pool to choose from
cloning_injected = 0
for idx in cloning_candidates:
    if cloning_injected >= n_cloning:
        break
    row = df.loc[idx]
    cid = row["customer_id"]
    card = row["card_number"]
    t1 = pd.to_datetime(row["timestamp"])
    # choose remote merchant far away: ensure distance > 800 km to create impossible travel within delta_t ~ 1 hour
    far_merchants = merch_df.copy()
    # compute distance from this txn's merchant to all merchants; choose merchant where distance > 800 km
    lat1, lon1 = row["merchant_lat"], row["merchant_long"]
    far_merchants["dist_from_t1"] = far_merchants.apply(lambda r: haversine(lat1, lon1, r["merchant_lat"], r["merchant_long"]), axis=1)
    far_candidates = far_merchants[far_merchants["dist_from_t1"] > 800]
    if far_candidates.empty:
        continue
    target_merch = far_candidates.sample(n=1, random_state=SEED+cloning_injected).iloc[0]
    # create a new transaction t2 for same card within short delta (e.g., +30 minutes)
    delta_minutes = random.randint(5, 120)  # small delta; but distance ensures impossible speed
    t2 = t1 + pd.Timedelta(minutes=delta_minutes)
    # find an index to modify or append new? We'll modify a different row from df to simulate concurrent usage
    # pick another row index for this customer or any random row to overwrite with impossible travel usage
    other_idx = df.sample(n=1, random_state=SEED+100+cloning_injected).index[0]
    df.at[other_idx, "customer_id"] = cid
    df.at[other_idx, "card_number"] = card
    df.at[other_idx, "merchant_id"] = target_merch["merchant_id"]
    df.at[other_idx, "merchant_category"] = target_merch["merchant_category"]
    df.at[other_idx, "merchant_lat"] = target_merch["merchant_lat"]
    df.at[other_idx, "merchant_long"] = target_merch["merchant_long"]
    df.at[other_idx, "timestamp"] = t2.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.at[other_idx, "ts_dt"] = pd.to_datetime(df.at[other_idx, "timestamp"])
    df.at[other_idx, "hour"] = df.at[other_idx,"ts_dt"].hour
    df.at[other_idx, "day_of_week"] = df.at[other_idx,"ts_dt"].weekday()
    df.at[other_idx, "month"] = df.at[other_idx,"ts_dt"].month
    df.at[other_idx, "amount"] = round(max(50.0, float(np.exp(np.random.normal(*amt_params[target_merch["merchant_category"]])))),2)
    df.at[other_idx, "distance_from_home"] = round(haversine(df.at[other_idx,"home_lat"], df.at[other_idx,"home_long"], df.at[other_idx,"merchant_lat"], df.at[other_idx,"merchant_long"]),2)
    # Mark as fraud
    df.at[other_idx, "is_fraud"] = 1
    df.at[other_idx, "fraud_type"] = "card_cloning"
    cloning_injected += 1
    # Optionally make the original row also fraud (concurrent) with same timestamp (simulate clone used simultaneously)
    if random.random() < 0.3:
        df.at[idx, "is_fraud"] = 1
        df.at[idx, "fraud_type"] = "card_cloning"
        cloning_injected += 1
    remaining_to_inject = target_frauds - df["is_fraud"].sum()

# --- Inject Account Takeover (ATO) patterns: choose customers and boost frequency & amounts in a short window
ato_customers = df["customer_id"].drop_duplicates().sample(n= int(0.01 * N_CUSTOMERS), random_state=SEED+2).tolist()  # ~1% customers
ato_injected = 0
for cust in ato_customers:
    if ato_injected >= n_ato:
        break
    # pick a compromise time in period (as UTC-aware datetime)
    compromise_time = (START_DATE + timedelta(seconds=random.randint(0, total_seconds))).replace(tzinfo=timezone.utc)
    # find transactions for this customer near compromise_time; if not enough, sample some and convert
    cust_idxs = df[df["customer_id"] == cust].index.tolist()
    if len(cust_idxs) < 3:
        continue
    # pick 3-8 transactions for this customer to convert into ATO burst
    k = random.randint(3, 8)
    chosen = random.sample(cust_idxs, min(k, len(cust_idxs)))
    # escalate amounts and times to be within short window
    base_increase = random.uniform(3.0, 10.0)  # scale multiplier
    for j, idx in enumerate(chosen):
        # set timestamp to be sequential after compromise_time
        new_ts = compromise_time + timedelta(minutes=j * random.randint(1,10))
        df.at[idx, "timestamp"] = new_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.at[idx, "ts_dt"] = pd.to_datetime(df.at[idx,"timestamp"])
        df.at[idx, "hour"] = df.at[idx,"ts_dt"].hour
        df.at[idx, "day_of_week"] = df.at[idx,"ts_dt"].weekday()
        df.at[idx, "month"] = df.at[idx,"ts_dt"].month
        # bump amount
        old_amt = df.at[idx,"amount"]
        new_amt = round(old_amt * base_increase * (1 + np.random.normal(0, 0.2)), 2)
        # cap new_amt reasonably
        df.at[idx,"amount"] = float(max(100.0, min(new_amt, 150000.0)))
        # change merchant categories to high-value categories with some prob
        if random.random() < 0.7:
            new_cat = random.choice(["electronics","jewelry","luxury_goods"])
            # pick a merchant of that category
            mlist = merchant_by_category[new_cat]
            if mlist:
                mid = random.choice(mlist)
                merch_row = merch_df[merch_df["merchant_id"] == mid].iloc[0]
                df.at[idx,"merchant_id"] = mid
                df.at[idx,"merchant_category"] = new_cat
                df.at[idx,"merchant_lat"] = merch_row["merchant_lat"]
                df.at[idx,"merchant_long"] = merch_row["merchant_long"]
                df.at[idx,"distance_from_home"] = round(haversine(df.at[idx,"home_lat"], df.at[idx,"home_long"], df.at[idx,"merchant_lat"], df.at[idx,"merchant_long"]),2)
        # mark as fraud
        df.at[idx,"is_fraud"] = 1
        df.at[idx,"fraud_type"] = "account_takeover"
        ato_injected += 1
        if ato_injected >= n_ato:
            break
    remaining_to_inject = target_frauds - df["is_fraud"].sum()

# --- Velocity probes: small repeated transactions (card testing) that sometimes are fraud
# For a small set of cards, create rapid-fire low-amount txns and mark some as fraud
probe_cards = df["card_number"].drop_duplicates().sample(n=100, random_state=SEED+3).tolist()
probe_injected = 0
for card in probe_cards:
    if probe_injected >= 200:  # cap
        break
    # pick k small transactions indices to modify
    idxs = df[df["card_number"] == card].sample(n=min(10, df[df["card_number"]==card].shape[0]), random_state=SEED+4).index.tolist()
    # make them happen in rapid succession and small amounts
    base_time = (START_DATE + timedelta(seconds=random.randint(0, total_seconds))).replace(tzinfo=timezone.utc)
    for j, idx in enumerate(idxs):
        new_ts = base_time + timedelta(seconds=j * random.randint(5, 60))
        df.at[idx,"timestamp"] = new_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.at[idx,"ts_dt"] = pd.to_datetime(df.at[idx,"timestamp"])
        df.at[idx,"hour"] = df.at[idx,"ts_dt"].hour
        df.at[idx,"day_of_week"] = df.at[idx,"ts_dt"].weekday()
        df.at[idx,"month"] = df.at[idx,"ts_dt"].month
        df.at[idx,"amount"] = float(round(np.random.choice([10,20,50,99,199,499]) * (1 + np.random.normal(0,0.05)),2))
        # mark some of these probes as fraud (~30%)
        if random.random() < 0.3 and df.at[idx,"is_fraud"] == 0:
            df.at[idx,"is_fraud"] = 1
            df.at[idx,"fraud_type"] = "card_cloning"
            probe_injected += 1
    remaining_to_inject = target_frauds - df["is_fraud"].sum()

# --- Edge cases: fast-but-legit travelers and holiday spikes already somewhat present via urban centers and time sampling
# Add some holiday spending bursts (legitimate)
holiday_customers = df["customer_id"].drop_duplicates().sample(n=200, random_state=SEED+5).tolist()
for cust in holiday_customers:
    # find some rows and increase amounts but keep is_fraud=0
    idxs = df[df["customer_id"] == cust].sample(n=min(3, df[df["customer_id"]==cust].shape[0]), random_state=SEED+6).index.tolist()
    for idx in idxs:
        df.at[idx,"amount"] = round(df.at[idx,"amount"] * random.uniform(2.0, 5.0),2)
        # mark as legitimate (non-fraud)
        df.at[idx,"is_fraud"] = 0
        df.at[idx,"fraud_type"] = "none"

# Final adjustment: if we still haven't hit target fraud count, randomly flip additional legitimate rows to fraud (smart structurers and random)
current_frauds = int(df["is_fraud"].sum())
to_add = target_frauds - current_frauds
if to_add > 0:
    # pick candidate indices that are currently non-fraud and elevate them (prefer large amounts or odd hours)
    candidates = df[df["is_fraud"] == 0].copy()
    # rank by amount desc + odd hour score
    candidates["score"] = candidates["amount"] * (1 + ((candidates["hour"] < 6) * 1.5))
    chosen_idxs = candidates.sort_values("score", ascending=False).head(to_add).index.tolist()
    for idx in chosen_idxs:
        # set as fraud (mix types)
        typ = random.choices(["card_cloning","account_takeover","merchant_collusion"], weights=[0.45,0.40,0.15])[0]
        df.at[idx,"is_fraud"] = 1
        df.at[idx,"fraud_type"] = typ

# If we have too many frauds (unlikely), revert some random to non-fraud
current_frauds = int(df["is_fraud"].sum())
if current_frauds > target_frauds:
    to_remove = current_frauds - target_frauds
    fraud_idxs = df[df["is_fraud"] == 1].sample(n=to_remove, random_state=SEED+7).index.tolist()
    for idx in fraud_idxs:
        df.at[idx,"is_fraud"] = 0
        df.at[idx,"fraud_type"] = "none"

# Final housekeeping: fill final columns exactly as requested and drop helper cols
df["is_fraud"] = df["is_fraud"].astype(int)
# ensure transaction_id continuity
df["transaction_id"] = [f"TXN_{i:08d}" for i in range(1, len(df)+1)]
# ensure timestamp iso format
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
# ensure required columns in order
final_cols = [
    "transaction_id", "customer_id", "card_number", "timestamp", "amount",
    "merchant_id", "merchant_category", "merchant_lat", "merchant_long",
    "is_fraud", "fraud_type", "hour", "day_of_week", "month", "distance_from_home"
]
# compute distance_from_home final (already present but ensure consistency)
df["distance_from_home"] = df.apply(lambda r: round(haversine(r["home_lat"], r["home_long"], r["merchant_lat"], r["merchant_long"]),2), axis=1)

out_df = df[final_cols].copy()

# sanity checks
assert len(out_df) == N_ROWS
fraud_count = int(out_df["is_fraud"].sum())
print(f"Generated {N_ROWS} rows; fraud_count = {fraud_count} (target {target_frauds})")

# save CSV
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out_df.to_csv(OUT_PATH, index=False)
print(f"Wrote dataset to {OUT_PATH}")
