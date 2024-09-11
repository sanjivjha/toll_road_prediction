# Toll Road Prediction Project

This project implements a machine learning solution to predict whether a vehicle is on a toll road, estimate the distance traveled on the toll road, and calculate the potential toll charges. It uses a combination of Graph Neural Networks (GNN) and Transformer models, trained on synthetic data generated from OpenStreetMap (OSM) information.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

The Toll Road Prediction system uses road network data from OpenStreetMap to create a GNN model of the road network. It then generates synthetic trajectory data to train a Transformer model for making predictions about toll road usage, distance, and charges.

## Features

- Data collection from OpenStreetMap
- Synthetic data generation for training
- Graph Neural Network (GNN) for road network representation
- Transformer model for toll road predictions
- Modular and testable code structure
- Deployment support for Amazon SageMaker

## Prerequisites

- Python 3.7+
- PyTorch 1.8+
- transformers library
- osmnx library
- networkx library
- AWS account (for SageMaker deployment)

## Project Structure

```
toll_road_prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── osm_data_collector.py
│   │   └── synthetic_data_generator.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_model.py
│   │   └── transformer_model.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── gnn_trainer.py
│   │   └── transformer_trainer.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py
│
├── tests/
│   ├── test_data_collection.py
│   ├── test_synthetic_data.py
│   ├── test_gnn_model.py
│   ├── test_transformer_model.py
│   └── test_inference.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/toll-road-prediction.git
   cd toll-road-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Collect OSM data:
   ```
   python src/data/osm_data_collector.py
   ```

2. Generate synthetic data:
   ```
   python src/data/synthetic_data_generator.py
   ```

3. Train the GNN model:
   ```
   python src/training/gnn_trainer.py
   ```

4. Train the Transformer model:
   ```
   python src/training/transformer_trainer.py
   ```

5. Run inference:
   ```
   python src/inference/predictor.py
   ```

## Testing

Run the unit tests using:

```
python -m unittest discover tests
```

## Deployment

To deploy the model on Amazon SageMaker:

1. Package the model:
   ```
   mkdir model
   cp src/models/transformer_model.py model/
   cp src/inference/predictor.py model/
   cp path/to/saved/model.pth model/
   tar -czvf model.tar.gz model
   ```

2. Upload the model to S3:
   ```
   aws s3 cp model.tar.gz s3://your-bucket/model.tar.gz
   ```

3. Use the SageMaker deployment script provided in the project to deploy the model.

4. Test the deployed endpoint using the provided code snippet in the deployment instructions.

For detailed deployment instructions, refer to the [Deployment section](#deployment) in the project documentation.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more detailed information about each component, please refer to the individual module documentation within the `src/` directory.

If you encounter any issues or have questions, please open an issue on the GitHub repository.# toll_road_prediction
This repository contain the code for prediction model if a vehicle is on toll road or not. 
