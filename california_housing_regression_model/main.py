import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Utils:
    @staticmethod
    def preprocess_data(data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        transformed = scaler.fit_transform(data[['longitude', 'latitude', 'housing_median_age',
                                                 'total_rooms',
                                                 'total_bedrooms', 'population', 'households',
                                                 'median_income', 'median_house_value']])
        columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                   'total_bedrooms', 'population', 'households', 'median_income',
                   'median_house_value']
        for i, col in enumerate(columns):
            data[col] = transformed[:, i]
        return data

    @staticmethod
    def convert_to_tensors(data):
        x = data.drop('median_house_value', axis=1).to_numpy()
        y = data['median_house_value'].to_numpy()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


class HousingModel:
    def __init__(self, data_path, optimizer_type, lr=0.01, batch_size=64, epochs=100):
        self.data_path = data_path
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_dim = 8

        # Placeholders for data
        self.train_data = None
        self.test_data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.optimizer = None
        self.loss_function = nn.MSELoss()

    def load_and_process_data(self):
        # load and clean data
        data = pd.read_csv(self.data_path)
        data = data.dropna(subset=['total_bedrooms'])
        data = data.drop(columns=['ocean_proximity'])

        # split data to train and test sets
        self.train_data, self.test_data = train_test_split(data,
                                                           test_size=0.2,
                                                           random_state=22)
        print(f"Training set: {self.train_data.shape}")
        print(f"Testing set: {self.test_data.shape}")

        # reset index
        self.train_data.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)

        # scale data
        self.train_data = Utils.preprocess_data(self.train_data)
        self.test_data = Utils.preprocess_data(self.test_data)

        # prepare tensors
        self.x_train, self.y_train = Utils.convert_to_tensors(self.train_data)
        self.x_test, self.y_test = Utils.convert_to_tensors(self.test_data)


    def initialize_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.initialize_optimizer()
        print(f"Model initialized with optimizer: {self.optimizer_type}")

    def initialize_optimizer(self):
        if self.optimizer_type == 1:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == 2:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5, weight_decay=0.001)
        elif self.optimizer_type == 3:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.005)
        else:
            raise ValueError("Invalid optimizer type. Choose 1 for SGD, 2 for SGD with momentum, or 3 for Adam.")

    def train(self):
        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0.0

            for start_idx in range(0, len(self.x_train), self.batch_size):
                x_batch = self.x_train[start_idx:start_idx + self.batch_size]
                y_batch = self.y_train[start_idx:start_idx + self.batch_size]

                self.optimizer.zero_grad()
                predictions = self.model(x_batch).squeeze()
                loss = self.loss_function(predictions, y_batch.squeeze())
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            num_train_samples = len(self.x_train)
            average_train_loss = total_train_loss / num_train_samples
            train_losses.append(average_train_loss)

            # Evaluation phase
            self.model.eval()
            total_test_loss = 0.0

            with torch.no_grad():
                for start_idx in range(0, len(self.x_test), self.batch_size):
                    x_batch = self.x_test[start_idx:start_idx + self.batch_size]
                    y_batch = self.y_test[start_idx:start_idx + self.batch_size]

                    predictions = self.model(x_batch).squeeze()
                    loss = self.loss_function(predictions, y_batch.squeeze())
                    total_test_loss += loss.item()

            num_test_samples = len(self.x_test)
            average_test_loss = total_test_loss / num_test_samples
            test_losses.append(average_test_loss)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {average_train_loss:.7f}, Test Loss: {average_test_loss:.7f}")

        return train_losses, test_losses


    def plot_losses(self, train_losses, test_losses):
        plt.figure(figsize=(6, 4))
        plt.plot(range(self.epochs), train_losses, label='Train Loss', color='blue')
        plt.plot(range(self.epochs), test_losses, label='Test Loss', color='red')
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()



DATA_PATH = "housing.csv"
OPTIMIZER_TYPE = 2  # 1: SGD, 2: SGD with Momentum, 3: Adam

# Initialize and run the model
housing_model = HousingModel(DATA_PATH, optimizer_type=OPTIMIZER_TYPE)
housing_model.load_and_process_data()
housing_model.initialize_model()
train_losses, test_losses = housing_model.train()
housing_model.plot_losses(train_losses, test_losses)
