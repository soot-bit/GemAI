from utils import load_config, get_path, get_data

def main():
    config = load_config()
    
    # Prepare data
    train_df, test_df, _ = get_data(config)
    
    # Save processed data
    train_df.to_parquet(get_path("train_data", config))
    test_df.to_parquet(get_path("test_data", config))
    
    print(f"✅ Data processed. Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Data saved to: {get_path('processed_dir', config)}")

if __name__ == "__main__":
    main()
