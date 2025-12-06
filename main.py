import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import re
from collections import Counter
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_LEN = 40
BATCH_SIZE = 4
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 5
LEARNING_RATE = 0.0001
EPOCHS = 15
VOCAB_SIZE = 14000
MAX_SEQ_LENGTH = 150

import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
import re
from datasets import load_dataset
import numpy as np

class SSTDataset(Dataset):
    def __init__(self, sentences, labels, word2idx, max_len=50):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        tokens = re.findall(r"\b\w+(?:[-']\w+)*\b|[!?.;,]", sentence.lower())
        indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in tokens]

        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.word2idx['<pad>']] * (self.max_len - len(indices))

        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data():

    try:
        ds = load_dataset("mattbit/tweet-sentiment-airlines")
        print("Доступные splits:", list(ds.keys()))

        print("\nСтруктура примера train:")
        print(ds['train'][0])

    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

    train_sentences, train_labels = [], []
    val_sentences, val_labels = [], []

    if 'train' in ds:
        print(f"\nОбработка train split: {len(ds['train'])} примеров")
        for example in ds['train']:
            if 'Text' in example and 'label' in example:
                train_sentences.append(example['Text'])
                train_labels.append(example['label'])
            elif 'text' in example and 'label' in example:
                train_sentences.append(example['text'])
                train_labels.append(example['label'])

    if 'test' in ds:
        print(f"Обработка test split как validation: {len(ds['test'])} примеров")
        for example in ds['test']:
            if 'Text' in example and 'label' in example:
                val_sentences.append(example['Text'])
                val_labels.append(example['label'])
            elif 'text' in example and 'label' in example:
                val_sentences.append(example['text'])
                val_labels.append(example['label'])

    if not val_sentences and len(train_sentences) > 1000:
        from sklearn.model_selection import train_test_split
        train_sentences, val_sentences, train_labels, val_labels = train_test_split(
            train_sentences, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )

    print(f"Train: {len(train_sentences)} примеров")
    print(f"Validation: {len(val_sentences)} примеров")

    if train_labels:
        print(f"\nТипы меток в train: {set(train_labels)}")
        print(f"Распределение классов в train:")
        unique, counts = np.unique(train_labels, return_counts=True)
        for cls, count in zip(unique, counts):
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment_name = sentiment_map.get(cls, f'unknown_{cls}')
            print(f"  {sentiment_name}: {count} ({count/len(train_labels):.1%})")


    print("\nСоздание словаря...")
    word_counts = Counter()

    for sentence in train_sentences:
        if isinstance(sentence, str):
            tokens = re.findall(r"\b\w+(?:[-']\w+)*\b|[!?.;,]", sentence.lower())
            word_counts.update(tokens)

    print(f"Всего уникальных слов: {len(word_counts)}")
    if word_counts:
        print(f"10 самых частых слов: {word_counts.most_common(10)}")
    else:
        print("ОШИБКА: Нет слов для подсчета!")
        return None

    vocab = ['<pad>', '<unk>'] + [word for word, count in word_counts.most_common(VOCAB_SIZE - 2)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    total_words = sum(word_counts.values())
    if total_words > 0:
        covered_words = sum(count for word, count in word_counts.most_common(VOCAB_SIZE) if word in word2idx)
        coverage = covered_words / total_words
        print(f"Размер словаря: {len(vocab)}")
        print(f"Покрытие словаря: {coverage:.2%}")
    else:
        print("ОШИБКА: Нет слов для создания словаря!")
        return None

    return (train_sentences, train_labels,
            val_sentences, val_labels,
            word2idx, len(vocab))

result = load_and_preprocess_data()

if result is None:
    print("Не удалось загрузить данные!")
else:
    (train_sentences, train_labels,
     val_sentences, val_labels,
     word2idx, vocab_size) = result

    train_dataset = SSTDataset(train_sentences, train_labels, word2idx, MAX_LEN)
    val_dataset = SSTDataset(val_sentences, val_labels, word2idx, MAX_LEN)

    print(f"Train dataset: {len(train_dataset)} примеров")
    print(f"Val dataset: {len(val_dataset)} примеров")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {MAX_LEN}")

    print(f"\nПроверка примеров:")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        original_text = train_sentences[i][:50] + "..." if len(train_sentences[i]) > 50 else train_sentences[i]
        label_names = ['negative', 'neutral', 'positive']
        label_value = sample['label'].item()
        label_name = label_names[label_value] if 0 <= label_value < len(label_names) else f'unknown_{label_value}'

        print(f"Пример {i+1}:")
        print(f"  Текст: '{original_text}'")
        print(f"  Метка: {label_value} ({label_name})")
        print(f"  Токены shape: {sample['text'].shape}")
        print(f"  Не-pad токены: {(sample['text'] != word2idx['<pad>']).sum().item()}")
        print()

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_filters=128, output_dim=3, dropout=0.3):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        self.conv1d_1 = nn.Conv1d(embed_dim, n_filters, 3, padding=1)
        self.conv1d_2 = nn.Conv1d(embed_dim, n_filters, 4, padding=1)
        self.conv1d_3 = nn.Conv1d(embed_dim, n_filters, 5, padding=1)

        self.batch_norm1 = nn.BatchNorm1d(n_filters)
        self.batch_norm2 = nn.BatchNorm1d(n_filters)
        self.batch_norm3 = nn.BatchNorm1d(n_filters)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * 3, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        embedded = embedded.permute(0, 2, 1)

        conv1 = F.relu(self.batch_norm1(self.conv1d_1(embedded)))
        conv2 = F.relu(self.batch_norm2(self.conv1d_2(embedded)))
        conv3 = F.relu(self.batch_norm3(self.conv1d_3(embedded)))

        pooled1 = F.adaptive_max_pool1d(conv1, 1).squeeze(2)
        pooled2 = F.adaptive_max_pool1d(conv2, 1).squeeze(2)
        pooled3 = F.adaptive_max_pool1d(conv3, 1).squeeze(2)

        concatenated = torch.cat([pooled1, pooled2, pooled3], dim=1)
        concatenated = self.dropout(concatenated)

        output = self.fc(concatenated)
        return output

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=3, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.embed_dropout(embedded)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        context_vector = attn_out[:, -1, :]

        return self.fc(context_vector)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=3, n_layers=3, n_heads=4, dropout=0.2, max_length=128):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        self.embed_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim)
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )

        self.embed_dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)

    def _create_src_mask(self, src, pad_idx=0):
        src_mask = (src != pad_idx)
        return src_mask

    def forward(self, text):
        batch_size, seq_len = text.size()

        token_embedded = self.embedding(text)
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(text.device)
        pos_embedded = self.pos_embedding(positions)

        embedded = token_embedded + pos_embedded
        embedded = self.embed_norm(embedded)
        embedded = self.embed_dropout(embedded)

        src_key_padding_mask = (text == 0)

        encoded = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )

        attention_weights = F.softmax(self.attention_pool(encoded).squeeze(-1), dim=1)
        attention_weights = attention_weights * (text != 0).float()
        attention_weights = F.normalize(attention_weights, p=1, dim=1)

        attention_weights = attention_weights.unsqueeze(-1)
        context_vector = torch.sum(encoded * attention_weights, dim=1)

        context_vector = self.final_dropout(context_vector)
        output = self.classifier(context_vector)

        return output

def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in tqdm(iterator, desc="Training"):
        optimizer.zero_grad()
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        predictions = model(texts)
        loss = criterion(predictions, labels)

        acc = multiclass_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            predictions = model(texts)
            loss = criterion(predictions, labels)

            acc = multiclass_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), accuracy, precision, recall, f1, all_preds, all_labels

def multiclass_accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]]).to(device)

def plot_training_history(history, model_name):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training History - {model_name}', fontsize=16)

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(history['precision'], label='Precision', color='green')
    axes[0, 2].plot(history['recall'], label='Recall', color='orange')
    axes[0, 2].plot(history['f1'], label='F1-Score', color='red')
    axes[0, 2].set_title('Metrics')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    plt.tight_layout()
    plt.show()

dataset = SSTDataset(train_sentences, train_labels, word2idx, MAX_LENGTH)
train_dataset = SSTDataset(train_sentences, train_labels, word2idx, MAX_LENGTH)
val_dataset = SSTDataset(val_sentences, val_labels, word2idx, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Размер тренировочного набора: {len(train_dataset)}")
print(f"Размер валидационного набора: {len(val_dataset)}")

models = [
    (TextCNN(vocab_size, EMBEDDING_DIM, n_filters=128, output_dim=3, dropout=0.3), "TextCNN"),
    (TransformerClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, output_dim=3, n_layers=3, n_heads=4, dropout=0.2, max_length=MAX_LEN), "Transformer"),
    (BiLSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 3, N_LAYERS), "BiLSTM")
]

all_results = {}

for model, name in models:
    print(f"\n{'='*50}")
    print(f"Starting experiment for {name}")
    print(f"{'='*50}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'final_preds': [], 'final_true_labels': []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
        val_loss, val_acc, accuracy, precision, recall, f1, all_preds, all_labels = evaluate_model(
            model, val_loader, criterion
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['accuracy'].append(accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        if epoch == EPOCHS - 1:
            history['final_preds'] = all_preds
            history['final_true_labels'] = all_labels

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    all_results[name] = history

    plot_training_history(history, name)

print("\nСравнение финальных результатов всех моделей:")

final_metrics = []
for name, history in all_results.items():
    final_metrics.append({
        'Model': name,
        'Val Loss': history['val_loss'][-1],
        'Val Accuracy': history['accuracy'][-1],
        'Precision': history['precision'][-1],
        'Recall': history['recall'][-1],
        'F1-Score': history['f1'][-1]
    })

results_df = pd.DataFrame(final_metrics)
results_df = results_df.round(4)
print("\n" + results_df.to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Comparison', fontsize=16)

metrics_to_plot = ['Val Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i//2, i%2]
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
