import pandas as pd
import os

def download_real_dataset():
    print("⏳ Downloading massive real-world dataset from the internet...")
    print("This might take 10 to 30 seconds depending on your internet speed. Please wait...")
    
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
    
    try:
        df = pd.read_csv(url)
        
        df = df[['text', 'label']]
        
        df = df.dropna()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, 'dataset', 'news_dataset.csv')
        
        df.to_csv(dataset_path, index=False)
        
        print(f"✅ SUCCESS! Downloaded {len(df)} real news articles.")
        print(f"✅ Overwrote the old dummy dataset at: {dataset_path}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")

if __name__ == '__main__':
    download_real_dataset()