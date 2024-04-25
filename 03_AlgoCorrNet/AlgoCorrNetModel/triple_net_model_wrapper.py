


# standard library imports
import os
import time

# related third-party
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader

# local application/library specific imports
from APP_CONFIG import Config
from AlgoCorrNetModel.datatypes import TrainingMNRLFeature
from AlgoCorrNetModel.model_wrapper import ModelWrapper
from AlgoCorrNetModel.utils import format_time
from AlgoCorrNetModel.datasets import TripleInputTrainingDataset

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()


class TripleNetModelWrapper(ModelWrapper):

    def __init__(self, model, freeze_layers=False, rnd_seed=0):
        super().__init__(model, freeze_layers, rnd_seed)

        self.criterion = torch.nn.TripletMarginLoss()
        self.model_save_path = CONFIG['NetModel_PATH'] + 'TripleNetModel_Pytorch'
        self.model_save_path_checkpoint = self.model_save_path + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'

    def __set_training_data_loaders(self):
        """
        Create data loaders for training dataset.

        This method creates data loaders for the training dataset using the configured batch size, and a linear learning rate scheduler with warm-up steps.
        The data loaders are stored in the class attribute `self.train_dataloader` for later use during training.

        Returns:
            None
        """

        # Create the training dataset from the dataframe
        self.train_dataset = TripleInputTrainingDataset(self.train_dataframe)

        # Create a random number generator with the global seed
        random_generator = torch.Generator().manual_seed(self.seed)

        # Create a random sampler based on training dataset and the generator
        train_sampler = RandomSampler(self.train_dataset, generator=random_generator)

        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batches,
            num_workers=0,
            pin_memory=True
        )

        # Total number of training steps is [number of batches] x [number of epochs]
        num_training_steps = len(train_dataloader) * self.epochs

        # Setup warmup for first ~10% of steps
        warmup_steps = int(0.1 * num_training_steps)

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        self.train_dataloader = train_dataloader

    def train(self, train_dataframe, validation_correlation_dataframe, validation_precision_at_k_dataframe, epochs=1,
              batches=1):
        """
        Train the model using the provided training data.
        This method performs model training using the provided training data and validation data.
        Additionally, it computes and prints the average training loss for each epoch, and computes and stores precision-at-k and correlation scores for validation data after each epoch.

        Args:
            train_dataframe (pandas.DataFrame): The training data as a Pandas DataFrame.
            validation_correlation_dataframe (pandas.DataFrame): The validation data for computing correlation scores as
                a Pandas DataFrame.
            validation_precision_at_k_dataframe (pandas.DataFrame): The validation data for computing precision-at-k
                scores as a Pandas DataFrame.
            epochs (int, optional): The number of epochs to train the model. Defaults to 1.
            batches (int, optional): The number of batches to use during training. Defaults to 1.
        Returns:
            None
        """

        assert train_dataframe is not None, "train_dataframe cannot be None"
        assert validation_correlation_dataframe is not None, "validation_correlation_dataframe cannot be None"
        assert validation_precision_at_k_dataframe is not None, "validation_precision_at_k_dataframe cannot be None"

        self.epochs = epochs
        self.batches = batches
        self.train_dataframe = train_dataframe
        self.__set_training_data_loaders()

        start_time = time.time()
        scaler = GradScaler()

        for epoch_i in range(self.start_epoch, self.epochs):
            print(f'======== Epoch {epoch_i + 1} / {self.epochs} ========')
            self.model.train()

            total_epoch_train_loss = 0

            for stp, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):

                self.model.zero_grad()
                for feature in TrainingMNRLFeature:
                    for batch_encoding in batch[feature.value]:
                        batch[feature.value][batch_encoding] = batch[feature.value][batch_encoding].to(self.device)

                anchor_text = batch[TrainingMNRLFeature.ANCHOR.value]
                positive_text = batch[TrainingMNRLFeature.POSITIVE.value]
                negative_text = batch[TrainingMNRLFeature.NEGATIVE.value]

                with autocast():
                    anchor_embeddings, positive_embeddings, negative_embedding = self.model(anchor_text, positive_text,
                                                                                            negative_text)

                    loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embedding)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()

                total_epoch_train_loss += loss.cpu().item()
            # all steps of an epoch end

            print(f'Epoch {epoch_i + 1}/{self.epochs}: loss = {total_epoch_train_loss / len(self.train_dataloader)}')
            self.save_model(epoch_i)
            # 1 epoch end

            print("***** Starting model validation *****")
            # start epoch validation
            self.compute_precision_at_k_scores(validation_precision_at_k_dataframe, validation_call=True)
            self.compute_correlation_scores(validation_correlation_dataframe, self.batches)
        # all epochs end

        print("***** Training complete! *****")
        print(f"Total training took {format_time(time.time() - start_time)}")
        self.isTrained = True
        self.model.cpu()
        self.model.eval()

        # empty cuda buffers
        torch.cuda.empty_cache()

        return None
