# pylint: disable=all

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
import torchvision


class SaguiDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        with open(annotation_file, "r") as f:
            self.coco_annotations = json.load(f)

        self.image_ids = {}
        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco_annotations["categories"]
        }

        for idx, img in enumerate(self.coco_annotations["images"]):
            self.image_ids[idx] = img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_info = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, img_info["file_name"])

        image = torchvision.io.read_image(img_path)
        image = image.float() / 255.0

        annotations = [
            ann
            for ann in self.coco_annotations["annotations"]
            if ann["image_id"] == img_info["id"]
        ]

        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target


def get_transform():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_model(train_loader, val_loader, num_classes, num_epochs=10):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_train_loss += losses.item()

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_val_loss += losses.item()

        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(epoch_val_loss / len(val_loader))

        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Model Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curves.png")
    plt.close()

    return model


def main():
    root_train = "..."
    root_valid = "..."
    annotations_train = os.path.join(root_train, "_annotations.coco.json")
    annotations_valid = os.path.join(root_valid, "_annotations.coco.json")

    transform = get_transform()

    train_dataset = SaguiDataset(root_train, annotations_train, transform)
    valid_dataset = SaguiDataset(root_valid, annotations_valid, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    num_classes = len(train_dataset.categories) + 1
    trained_model = train_model(train_loader, valid_loader, num_classes)

    torch.save(trained_model.state_dict(), "fasterrcnn_marmoset_detector.pth")
    torch.save(trained_model, "fasterrcnn_marmoset_detector.pt")


if __name__ == "__main__":
    main()
