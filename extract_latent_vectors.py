import pickle

with open("profiles.pkl", "rb") as f:
    profiles_dict = pickle.load(f)

print(f"Total profiles: {len(profiles_dict)}")

for i, (station_id, profile) in enumerate(profiles_dict.items()):
    print(f"{station_id}: type = {type(profile)}")
    if isinstance(profile, dict):
        print("Keys:", list(profile.keys()))
        for k in profile:
            print(f"  {k}: {type(profile[k])}, sample = {profile[k][:5] if hasattr(profile[k], '__getitem__') else profile[k]}")
    else:
        print("Sample:", profile[:5] if hasattr(profile, '__getitem__') else profile)

    if i == 5:
        break
