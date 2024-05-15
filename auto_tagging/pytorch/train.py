import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import MTATDataset
from hardmodel import ResCnn
import config
from dataset import mel_spectrogram


if __name__ == '__main__':
    print('USING DEVICE: {}'.format(config.DEVICE))
    # create dataset
    MTAT = MTATDataset(
        device=config.DEVICE,
        data_dir=config.DATA_DIR,
        annotations_file=config.ANNOTATIONS_FILE,
        tags_file=config.TAGS_FILE,
        transformation=mel_spectrogram,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
        target_image_size=config.IMAGE_SIZE,
    )
    print('THERE ARE {} SAMPLES IN DATASET'.format(len(MTAT)))
    # spliting dataset
    train_size = int(config.TRAIN_SIZE_PROPORTION * len(MTAT))
    test_size = int(config.TEST_SIZE_PROPORTION * len(MTAT))
    val_size = len(MTAT) - (train_size + test_size)
    train_dataset, test_dataset, val_dataset = random_split(
        MTAT, [train_size, test_size, val_size]
    )

    # creating dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    #create nn
    model = ResCnn(
        in_channels=config.IN_CHANNELS, 
        classes_num=config.CLASSES_NUM, 
        kernel_size=config.KERNEL_SIZE,
        stride=config.STRIDE,
        image_size=config.IMAGE_SIZE
    ).to(config.DEVICE)
    
    # choose optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()


    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir="runs/fashion_trainer_{}".format(timestamp))
    epoch_number = 0
    best_vloss = 1_000_000.0

    for epoch in range(config.EPOCHS_NUM):
        print("EPOCH {} / {}:".format(epoch_number + 1,config.EPOCHS_NUM))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "../models/version1/model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
