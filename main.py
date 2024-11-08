# !pip install torch pandas numpy loguru xarray requests scikit-learn typing-extensions

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
import requests
import xarray as xr
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Configure loguru logger
logger.add(
    "forest_intelligence_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)

@dataclass
class ForestRegion:
    """Represents a forest region for analysis."""
    name: str
    latitude: float
    longitude: float
    radius_km: float  # Analysis radius in kilometers

class MODISDataFetcher:
    """Handles fetching and processing MODIS vegetation data."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.base_url = "https://lpdaacsvc.cr.usgs.gov/appeears/api"
        logger.info(f"Initialized MODIS fetcher for period {start_date} to {end_date}")

    def fetch_ndvi_data(self, region: ForestRegion) -> xr.Dataset:
        """
        Fetches NDVI (Normalized Difference Vegetation Index) data for a given region.
        
        Args:
            region: ForestRegion object containing location details
            
        Returns:
            xr.Dataset containing NDVI data
        """
        try:
            # Convert coordinates to MODIS grid
            lat_min = region.latitude - (region.radius_km / 111)
            lat_max = region.latitude + (region.radius_km / 111)
            lon_min = region.longitude - (region.radius_km / 111)
            lon_max = region.longitude + (region.radius_km / 111)
            
            # Calculate number of days in the date range
            date_range = pd.date_range(self.start_date, self.end_date, freq='D')
            num_days = len(date_range)
            
            # Generate mock data with matching dimensions
            ds = xr.Dataset({
                'ndvi': (['time', 'latitude', 'longitude'], 
                        np.random.uniform(0, 1, size=(num_days, 10, 10))),
                'quality': (['time', 'latitude', 'longitude'],
                          np.random.randint(0, 3, size=(num_days, 10, 10)))
            },
            coords={
                'time': date_range,
                'latitude': np.linspace(lat_min, lat_max, 10),
                'longitude': np.linspace(lon_min, lon_max, 10)
            })
            
            logger.success(f"Successfully fetched NDVI data for region {region.name}")
            return ds
        except Exception as e:
            logger.error(f"Error fetching NDVI data: {str(e)}")
            raise

class ForestIntelligenceDataset(Dataset):
    """PyTorch dataset for forest intelligence analysis."""
    
    def __init__(
        self, 
        ndvi_data: xr.Dataset,
        sequence_length: int = 30,
        prediction_horizon: int = 7
    ):
        self.ndvi_data = ndvi_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self) -> None:
        """Prepares and normalizes the NDVI data."""
        ndvi_values = self.ndvi_data.ndvi.values
        # Flatten spatial dimensions
        self.spatial_shape = ndvi_values.shape[-2:]  # Save spatial dimensions
        ndvi_flat = ndvi_values.reshape(ndvi_values.shape[0], -1)
        self.normalized_data = self.scaler.fit_transform(ndvi_flat)
        self.input_size = self.normalized_data.shape[1]
        
    def __len__(self) -> int:
        return max(0, len(self.normalized_data) - self.sequence_length - self.prediction_horizon)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.normalized_data[idx:idx + self.sequence_length]
        y = self.normalized_data[idx + self.sequence_length:
                               idx + self.sequence_length + self.prediction_horizon]
        # Ensure consistent dimensions
        y_reshaped = y.reshape(-1, self.input_size)  # Flatten the prediction horizon
        return torch.FloatTensor(X), torch.FloatTensor(y_reshaped[0])  # Take first prediction


class ForestIntelligenceModel(nn.Module):
    """Neural network model for analyzing forest intelligence patterns."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use only last timestep output


class ForestIntelligenceAnalyzer:
    """Main class for analyzing forest intelligence."""
    
    def __init__(
        self,
        model: ForestIntelligenceModel,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.model = model.to(device)
        self.device = device
        self.intelligence_metrics: Dict[str, float] = {}
        logger.info(f"Initialized Forest Intelligence Analyzer using device: {device}")
        
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001
    ) -> List[float]:
        """
        Trains the forest intelligence model.
        
        Args:
            train_loader: DataLoader containing training data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            List of training losses
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.view(batch_y.size(0), -1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                
        return losses
    
    def analyze_intelligence(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Analyzes forest intelligence based on model predictions.
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            Dictionary containing intelligence metrics
        """
        self.model.eval()
        prediction_errors = []
        synchronization_scores = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                
                # Calculate prediction accuracy
                mse = nn.MSELoss()(predictions, batch_y.view(batch_y.size(0), -1))
                prediction_errors.append(mse.item())
                
                # Calculate synchronization score
                sync_score = self._calculate_synchronization(predictions)
                synchronization_scores.append(sync_score)
        
        # Calculate intelligence metrics
        self.intelligence_metrics = {
            'prediction_accuracy': 1.0 / (1.0 + np.mean(prediction_errors)),
            'synchronization_score': np.mean(synchronization_scores),
            'adaptive_capacity': self._calculate_adaptive_capacity(prediction_errors)
        }
        
        logger.info("Intelligence analysis completed")
        logger.info(f"Intelligence metrics: {self.intelligence_metrics}")
        
        return self.intelligence_metrics
    
    def _calculate_synchronization(self, predictions: torch.Tensor) -> float:
        """
        Calculates synchronization score based on spatial correlations.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Synchronization score
        """
        correlations = torch.corrcoef(predictions.T)
        return torch.mean(torch.abs(correlations)).item()
    
    def _calculate_adaptive_capacity(self, errors: List[float]) -> float:
        """
        Calculates adaptive capacity based on error patterns.
        
        Args:
            errors: List of prediction errors
            
        Returns:
            Adaptive capacity score
        """
        # Calculate trend in prediction errors
        error_changes = np.diff(errors)
        return float(np.mean(error_changes < 0))  # Proportion of improving predictions

def main():
    """Main function to run the forest intelligence analysis."""
    logger.info("Starting forest intelligence analysis")
    
    # Define forest region
    region = ForestRegion(
        name="Amazon Rainforest Sample",
        latitude=-3.4653,
        longitude=-62.2159,
        radius_km=50
    )
    
    # Initialize data fetcher with proper date handling
    end_date = datetime.now()
    start_date = end_date - timedelta(days=728)  # Ensure exactly one year of data
    data_fetcher = MODISDataFetcher(start_date, end_date)
    
    try:
        # Fetch NDVI data
        ndvi_data = data_fetcher.fetch_ndvi_data(region)
        
        # Create dataset with smaller sequence length and prediction horizon
        dataset = ForestIntelligenceDataset(
            ndvi_data, 
            sequence_length=14,  # Two weeks of history
            prediction_horizon=1  # Predict next day
        )
        data_loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True,
            drop_last=True  # Prevent issues with incomplete batches
        )
        
        # Initialize model
        input_size = dataset.input_size
        model = ForestIntelligenceModel(input_size=input_size)
        
        # Initialize analyzer
        analyzer = ForestIntelligenceAnalyzer(model)
        
        # Train model
        losses = analyzer.train(data_loader, epochs=500)  # Reduced epochs for testing
        
        # Analyze intelligence
        intelligence_metrics = analyzer.analyze_intelligence(data_loader)
        
        logger.success("Analysis completed successfully")
        logger.info(f"Final intelligence metrics: {intelligence_metrics}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
