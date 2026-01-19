import pandas as pd

# Fake News Detection Datasets from Kaggle used here to create a combined CSV. 
# Change true_path and fake_path when using other splitted datasets. 
# Remember that the splitted datasets should not be labeled before the merge.

true_path = 'dataset/News_dataset/True.csv'
fake_path = 'dataset/News_dataset/Fake.csv'

df_true = pd.read_csv(true_path)
df_fake = pd.read_csv(fake_path)

df_true['is_fake'] = 0
df_fake['is_fake'] = 1

df = pd.concat([df_true, df_fake], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("dataset/combined_fake_news.csv", index=False)

print("CSV combined created: combined_fake_news.csv")
print(df['is_fake'].value_counts())
