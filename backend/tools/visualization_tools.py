"""Visualization tools — create charts and plots."""

CREATE_VISUALIZATION_TOOL = {
    "name": "create_visualization",
    "description": (
        "Create a data visualization. Generates a matplotlib/seaborn chart and saves it as PNG. "
        "Specify the chart type, data source, and any customization."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "description": "Type of chart (bar, line, scatter, histogram, heatmap, box, pie, etc.)",
            },
            "dataset": {
                "type": "string",
                "description": "Dataset filename to visualize",
            },
            "x_column": {
                "type": "string",
                "description": "Column for x-axis",
            },
            "y_column": {
                "type": "string",
                "description": "Column for y-axis",
            },
            "title": {
                "type": "string",
                "description": "Chart title",
            },
            "code": {
                "type": "string",
                "description": "Custom matplotlib/seaborn code to execute for complex visualizations",
            },
        },
        "required": ["chart_type"],
    },
}
