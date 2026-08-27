import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from modules.erc import SequentialERC


class DailyDialogERC(Dataset):

    def __init__(self, text_file, emotion_file):

        self.samples = []

        with open(text_file, "r", encoding="utf-8") as f:
            texts = f.readlines()

        with open(emotion_file, "r", encoding="utf-8") as f:
            emotions = f.readlines()

        for text, emo in zip(texts, emotions):

            utterances = [
                x.strip()
                for x in text.split("__eou__")
                if x.strip()
            ]

            labels = [
                int(x)
                for x in emo.strip().split()
            ]

            if len(utterances) == len(labels):
                self.samples.append(
                    {
                        "texts": utterances,
                        "labels": labels
                    }
                )


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):

    texts = []
    labels = []

    for item in batch:

        dialogue = " </s> ".join(item["texts"])

        texts.append(dialogue)

        labels.extend(item["labels"])

    return {
        "texts": texts,
        "labels": torch.tensor(labels)
    }


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    model = SequentialERC()
    model.to(device)

    dataset = DailyDialogERC(
        "data/dailydialogue/train/dialogues_train.txt",
        "data/dailydialogue/train/dialogues_emotion_train.txt"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5
    )

    criterion = CrossEntropyLoss()

    model.train()

    for epoch in range(1):

        total_loss = 0

        for step, batch in enumerate(loader):

            labels = batch["labels"].to(device)

            output = model(
                {
                    "texts": batch["texts"]
                }
            )

            labels = labels[:output["logits"].size(0)]

            loss = criterion(
                output["logits"],
                labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    "step:",
                    step,
                    "loss:",
                    loss.item()
                )

        print(
            "epoch loss:",
            total_loss / len(loader)
        )


    os.makedirs(
        "pre_trained_models",
        exist_ok=True
    )

    model.save(
        "pre_trained_models/sequential_erc_model.pth"
    )

    print("saved")


if __name__ == "__main__":
    main()