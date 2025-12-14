"""
MULTI-DATASET LOADER FOR FAKE NEWS DETECTION
Combines LIAR, FakeNewsNet, and Kaggle datasets
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

class MultiDatasetLoader:
    """Load and combine multiple fake news datasets"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.datasets_loaded = {}

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def load_kaggle(self, fake_path='Fake.csv', real_path='True.csv', sample_size=None):
        """Load Kaggle Fake and Real News Dataset"""
        self._log("\n[1/3] Loading Kaggle Dataset...")

        try:
            fake_df = pd.read_csv(fake_path)
            real_df = pd.read_csv(real_path)

            fake_df['label'] = 1
            real_df['label'] = 0

            text_col = 'text' if 'text' in fake_df.columns else fake_df.columns[0]

            fake_texts = fake_df[text_col].dropna().tolist()
            real_texts = real_df[text_col].dropna().tolist()

            if sample_size:
                fake_texts = fake_texts[:sample_size]
                real_texts = real_texts[:sample_size]

            texts = fake_texts + real_texts
            labels = [1] * len(fake_texts) + [0] * len(real_texts)

            self._log(f"  ✓ Fake: {len(fake_texts)}, Real: {len(real_texts)}, Total: {len(texts)}")

            self.datasets_loaded['kaggle'] = {
                'texts': texts, 'labels': labels, 'size': len(texts),
                'fake': len(fake_texts), 'real': len(real_texts)
            }

            return texts, labels
        except FileNotFoundError:
            self._log(f"  ✗ Kaggle files not found")
            return [], []
        except Exception as e:
            self._log(f"  ✗ Error: {e}")
            return [], []

    def load_liar(self, train_path='train.tsv', valid_path='valid.tsv', test_path='test.tsv', sample_size=None):
        """Load LIAR Dataset"""
        self._log("\n[2/3] Loading LIAR Dataset...")

        texts = []
        labels = []
        real_count = 0
        fake_count = 0

        try:
            for tsv_file in [train_path, valid_path, test_path]:
                if not os.path.exists(tsv_file):
                    continue

                df = pd.read_csv(tsv_file, sep='\t', header=None)

                for idx, row in df.iterrows():
                    statement = str(row[2]).strip()
                    label_6class = str(row[1]).lower().strip()

                    if not statement or len(statement) < 10:
                        continue

                    if label_6class in ['true', 'mostly-true', 'half-true']:
                        label = 0
                        real_count += 1
                    else:
                        label = 1
                        fake_count += 1

                    texts.append(statement)
                    labels.append(label)

            if sample_size and texts:
                real_texts = [t for t, l in zip(texts, labels) if l == 0][:sample_size]
                fake_texts = [t for t, l in zip(texts, labels) if l == 1][:sample_size]
                texts = real_texts + fake_texts
                labels = [0] * len(real_texts) + [1] * len(fake_texts)

            if texts:
                self._log(f"  ✓ Fake: {fake_count}, Real: {real_count}, Total: {len(texts)}")
                self.datasets_loaded['liar'] = {'texts': texts, 'labels': labels, 'size': len(texts), 'fake': fake_count, 'real': real_count}
                return texts, labels
            else:
                return [], []
        except Exception as e:
            self._log(f"  ✗ Error: {e}")
            return [], []

    def load_fakenewsnet(self, politifact_fake_dir='politifact_fake', politifact_real_dir='politifact_real', gossipcop_fake_dir='gossipcop_fake', gossipcop_real_dir='gossipcop_real', sample_size=None):
        """Load FakeNewsNet Dataset"""
        self._log("\n[3/3] Loading FakeNewsNet Dataset...")

        texts = []
        labels = []
        real_count = 0
        fake_count = 0

        def load_json_files(directory, label):
            nonlocal texts, labels, real_count, fake_count
            count = 0
            if not os.path.exists(directory):
                return 0
            for file in Path(directory).glob('*.json'):
                try:
                    with open(file) as f:
                        data = json.load(f)
                        text = None
                        for key in ['text', 'content', 'body', 'article']:
                            if key in data and data[key]:
                                text = str(data[key]).strip()
                                break
                        if text and len(text) > 10:
                            texts.append(text)
                            labels.append(label)
                            count += 1
                            if label == 0:
                                real_count += 1
                            else:
                                fake_count += 1
                except:
                    continue
            return count

        try:
            load_json_files(politifact_fake_dir, 1)
            load_json_files(politifact_real_dir, 0)
            load_json_files(gossipcop_fake_dir, 1)
            load_json_files(gossipcop_real_dir, 0)

            if sample_size and texts:
                real_texts = [t for t, l in zip(texts, labels) if l == 0][:sample_size]
                fake_texts = [t for t, l in zip(texts, labels) if l == 1][:sample_size]
                texts = real_texts + fake_texts
                labels = [0] * len(real_texts) + [1] * len(fake_texts)

            if texts:
                self._log(f"  ✓ Fake: {fake_count}, Real: {real_count}, Total: {len(texts)}")
                self.datasets_loaded['fakenewsnet'] = {'texts': texts, 'labels': labels, 'size': len(texts), 'fake': fake_count, 'real': real_count}
                return texts, labels
            else:
                return [], []
        except Exception as e:
            self._log(f"  ✗ Error: {e}")
            return [], []

    def load_all_datasets(self, sample_size=None):
        """Load all 3 datasets and combine"""
        self._log("\n" + "="*70)
        self._log("LOADING ALL DATASETS")
        self._log("="*70)

        all_texts = []
        all_labels = []
        total_real = 0
        total_fake = 0

        try:
            texts, labels = self.load_kaggle(sample_size=sample_size)
            if texts:
                all_texts.extend(texts)
                all_labels.extend(labels)
                total_real += sum(1 for l in labels if l == 0)
                total_fake += sum(1 for l in labels if l == 1)
        except:
            pass

        try:
            texts, labels = self.load_liar(sample_size=sample_size)
            if texts:
                all_texts.extend(texts)
                all_labels.extend(labels)
                total_real += sum(1 for l in labels if l == 0)
                total_fake += sum(1 for l in labels if l == 1)
        except:
            pass

        try:
            texts, labels = self.load_fakenewsnet(sample_size=sample_size)
            if texts:
                all_texts.extend(texts)
                all_labels.extend(labels)
                total_real += sum(1 for l in labels if l == 0)
                total_fake += sum(1 for l in labels if l == 1)
        except:
            pass

        self._log("\n" + "="*70)
        self._log(f"Total Real: {total_real}, Total Fake: {total_fake}, Total: {len(all_texts)}")
        self._log("="*70 + "\n")

        return all_texts, all_labels

    def print_statistics(self):
        """Print statistics"""
        for dataset_name, data in self.datasets_loaded.items():
            print(f"{dataset_name.upper()}: Real={data['real']}, Fake={data['fake']}, Total={data['size']}")
