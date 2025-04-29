The Advanced Data Science AI Assistant is a powerful Streamlit web application that combines the capabilities of AI with data analysis tools to help users explore, analyze, and visualize their data through natural language queries. Powered by OpenAI's GPT models, this application enables users to perform complex data analysis tasks with simple text commands.
Features
🔍 Data Exploration

Upload custom datasets (CSV, Excel, JSON, Parquet)
Access pre-loaded sample datasets
View comprehensive data statistics and summaries
Explore missing values analysis and correlation matrices

🤖 AI-Powered Insights

Automatically generate key insights from your data
Detect anomalies using advanced machine learning techniques
Get suggested analysis questions based on your dataset
Perform automated Exploratory Data Analysis (EDA)

📊 Interactive Visualizations

Generate beautiful visualizations using Plotly (interactive) or Matplotlib/Seaborn (static)
Create custom plots through natural language requests
View visualization history and save favorites
Explore clustering and dimensionality reduction visualizations

💻 Code Generation

Convert natural language requests into Python code
Execute generated code directly in the app
Save and reuse code snippets
Copy code to clipboard for external use

📑 Export & Sharing

Generate comprehensive analysis reports
Export reports in Markdown, HTML formats
Download processed data in various formats (CSV, Excel, JSON, Parquet)
Save visualizations for presentations

Installation
Prerequisites

Python 3.8 or higher
An OpenAI API key

Setup

Clone the repository:

bashgit clone <repository-url>
cd advanced-data-science-assistant

Create a virtual environment:

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages:

bashpip install -r requirements.txt

Create a .env file in the project root directory:

OPENAI_API_KEY=your_openai_api_key_here

Run the application:

bashstreamlit run app.py
Usage Guide
Loading Data

Use the sidebar to upload your data file (CSV, Excel, JSON, or Parquet) or select a sample dataset
Once loaded, explore the data overview and statistics in the "Data Overview" tab

Generating Insights

Navigate to the "AI Insights" tab
Explore automatically generated insights about your data
View anomaly detection results and automated EDA findings

Creating Visualizations

Go to the "Generate Code" tab
Enter a natural language request like "Create a scatter plot of price vs. rating colored by category"
Select your preferred visualization library (Plotly or Matplotlib/Seaborn)
Click "Generate Code" to create the visualization code
Click "Execute Code" to run the code and display the visualization

Managing Visualizations

View your visualization history in the "Visualization Gallery" tab
Add useful visualizations to favorites for quick access
Re-execute any previous visualization with one click

Exporting Results

Navigate to the "Export & Share" tab
Generate a comprehensive report of your analysis
Download your processed data in various formats
Share your findings with colleagues

Example Queries

"Create an interactive scatter plot of price vs. rating colored by category"
"Show me the correlation between all numeric variables"
"Generate a box plot for each numeric column grouped by category"
"Create a time series plot of sales over time"
"Perform a cluster analysis on the numeric columns and visualize the clusters"
"Generate a heatmap of missing values"
"Create a 3D scatter plot of the first three principal components"

Technical Details
Libraries Used

Streamlit: For the web application framework
OpenAI API: For AI-powered code generation and insights
Pandas: For data manipulation and analysis
Plotly: For interactive visualizations
Matplotlib/Seaborn: For static visualizations
Scikit-learn: For machine learning algorithms (PCA, clustering, anomaly detection)
NumPy: For numerical operations

Architecture
The application follows a modular architecture with dedicated functions for:

Data loading and processing
AI code generation
Automated insights generation
Anomaly detection
Visualization rendering
Report generation
Code execution

Troubleshooting
Common Issues

OpenAI API Key Errors: Ensure your API key is correctly set in the .env file
File Upload Issues: Check if your file is in a supported format and is not corrupted
Memory Errors: Large datasets may cause memory issues; try using a subset of your data
Visualization Errors: Some complex visualizations may require specific data structures

Future Enhancements

Support for more complex data types (geospatial, text mining)
Integration with more AI models
Advanced statistical testing features
Custom visualization templates
Collaborative sharing features
PDF export functionality
User accounts and saved workspaces

License
MIT License
Acknowledgements

OpenAI for providing the GPT models
Streamlit for the web application framework
The open-source data science community for their invaluable tools and libraries
