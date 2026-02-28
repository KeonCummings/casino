"""Code interpreter tool — executes Python code and captures outputs."""

TOOL_DEFINITION = {
    "name": "execute_python",
    "description": (
        "Execute Python code to analyze data, create visualizations, train models, "
        "or perform computations. The code runs in a workspace with access to: "
        "pandas, numpy, matplotlib, seaborn, scikit-learn, scipy. "
        "Use plt.show() to save visualizations. "
        "Save datasets to DATASETS_DIR, visualizations to VISUALIZATIONS_DIR."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            }
        },
        "required": ["code"],
    },
}
