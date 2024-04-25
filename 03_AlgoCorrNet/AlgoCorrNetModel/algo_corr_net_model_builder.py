


# standard library imports
import os

# related third-party
import torch
from datetime import datetime

# local application/library specific imports
from APP_CONFIG import Config
from AlgoCorrNetModel.algo_corr_net_model import AlgoCorrNetModel
from AlgoCorrNetModel.base_net_model import BaseNetModel
from AlgoCorrNetModel.embedding_model import EmbeddingModel

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()


class AlgoCorrNetModelBuilder:
    def __init__(self):
        self.model_save_path = CONFIG['AlgoCorrNetModel_PATH'] + 'AlgoCorrNet-Pytorch'
        self.model_save_path_checkpoint = self.model_save_path + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
        self.model = None

    def __save_model(self):
        """
        Saves the model checkpoint.

        Args:
            None

        Returns:
            None
        """

        self.model_save_path_checkpoint = self.model_save_path + f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
        checkpoint = {
            'model_path': self.model_save_path_checkpoint,
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, self.model_save_path_checkpoint)
        print(f"Saved checkpoint {self.model_save_path_checkpoint}")

        return None

    def buildAlgoCorrNetModel(self, pretrained_model_path, save_model=False):
        """
        Build an AlgoCorrNetModel instance by loading a pretrained model from a checkpoint file.

        Args:
            pretrained_model_path (str): The path to the checkpoint file of the pretrained model.
            save_model (bool, optional): Whether to save the built model as a checkpoint file. Default is False.

        Returns:
            None
        """

        assert pretrained_model_path is not None, "pretrained_model_path cannot be None"

        embedding_model = EmbeddingModel()
        pretrained_model = BaseNetModel(embedding_model)
        checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        self.model = AlgoCorrNetModel(pretrained_model.embedding_model)
        self.model.eval()
        if save_model:
            self.__save_model()

        return None

    def loadAlgoCorrNetModel(self, model_path):
        """
        Loads a saved AlgoCorrNet model from a given model path.

        Args:
            model_path (str): The file path of the saved AlgoCorrNet model.

        Returns:
            AlgoCorrNetModel: The loaded AlgoCorrNetModel instance.
        """

        assert model_path is not None, "model_path cannot be None"

        embedding_model = EmbeddingModel()
        self.model = AlgoCorrNetModel(embedding_model)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.model_path = checkpoint['model_path']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        return self.model

    def getAlgoCorrNetModel(self):
        """
        Returns the loaded AlgoCorrNetModel model.

        Returns:
            AlgoCorrNetModel or None: The loaded AlgoCorrNet model, or None if the model is not available.
        """
        return self.model
