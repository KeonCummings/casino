"""Dataset management tools — list, describe, load datasets."""

LOAD_DATASET_TOOL = {
    "name": "load_dataset",
    "description": (
        "Load a well-known dataset (iris, titanic, wine, tips, penguins, etc.) "
        "or generate sample data for analysis. Saves the dataset to the workspace."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Dataset name (e.g., 'iris', 'titanic', 'tips') or 'generate' for synthetic data",
            },
            "description": {
                "type": "string",
                "description": "If generating, describe the data to create",
            },
        },
        "required": ["name"],
    },
}

LIST_DATASETS_TOOL = {
    "name": "list_datasets",
    "description": "List all datasets currently available in the workspace.",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

DESCRIBE_DATASET_TOOL = {
    "name": "describe_dataset",
    "description": "Get statistics and info about a dataset (shape, dtypes, describe(), null counts).",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The dataset filename to describe",
            }
        },
        "required": ["name"],
    },
}
