from typing import Tuple
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"File not found: {filepath}")
	return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	df_clean = df.copy()
	df_clean = df_clean.dropna()
	df_clean = df_clean.drop_duplicates()
	df_clean = df_clean.reset_index(drop=True)
	return df_clean


def scale_features(df: pd.DataFrame, target_col: str = 'tsunami') -> Tuple[pd.DataFrame, StandardScaler]:
	if target_col not in df.columns:
		raise ValueError(f"Target column '{target_col}' not found in dataframe")

	df_scaled = df.copy()
	# Pilih kolom numerik selain target
	numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
	if target_col in numeric_cols:
		numeric_cols.remove(target_col)

	scaler = StandardScaler()
	if numeric_cols:
		df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

	return df_scaled, scaler


def split_data(df: pd.DataFrame, target_col: str = 'tsunami', test_size: float = 0.2, random_state: int = 42):
	if target_col not in df.columns:
		raise ValueError(f"Target column '{target_col}' not found in dataframe")

	X = df.drop(columns=[target_col])
	y = df[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)
	return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath: str,
						target_col: str = 'tsunami',
						test_size: float = 0.2,
						random_state: int = 42,
						save_path: str = None):
	df = load_data(filepath)
	df = clean_data(df)
	df_scaled, scaler = scale_features(df, target_col=target_col)

	if save_path:
		df_scaled.to_csv(save_path, index=False)

	X_train, X_test, y_train, y_test = split_data(df_scaled, target_col=target_col, test_size=test_size, random_state=random_state)
	return X_train, X_test, y_train, y_test, df_scaled


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Preprocess earthquake/tsunami dataset')
	parser.add_argument('--input', '-i', default='earthquake_data_tsunami_raw.csv', help='Path to input CSV file')
	parser.add_argument('--target', '-t', default='tsunami', help='Target column name')
	parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
	parser.add_argument('--random-state', type=int, default=42, help='Random seed')
	parser.add_argument('--save', '-s', default='earthquake_data_tsunami_preprocessing.csv', help='Optional path to save scaled CSV')

	args = parser.parse_args()

	X_train, X_test, y_train, y_test, df_scaled = preprocess_pipeline(
		filepath=args.input,
		target_col=args.target,
		test_size=args.test_size,
		random_state=args.random_state,
		save_path=args.save
	)

	print(f"Preprocessing selesai. Data scaled saved to: {args.save}")
	print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")