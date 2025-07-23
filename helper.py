import pandas as pd
def f(path):
    df = pd.read_csv(path)
    df['label'] = 'body'
    df.to_csv(path, index=False)
    print(f"relabeled {path}")

if __name__ == "__main__":
    f(r'data/labeled_data/file20.csv')