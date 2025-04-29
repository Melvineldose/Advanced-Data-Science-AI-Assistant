import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import os
import sys
import re
import base64
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced Data Science AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1.5rem;
        text-align: center;
        background: linear-gradient(90deg, #1E88E5, #512DA8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 500;
        color: #424242;
        margin-bottom: 1rem;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.5rem;
    }
    
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    
    .stTextInput>div>div>input {
        padding: 1rem;
        font-size: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    
    .stTextArea>div>div>textarea {
        padding: 1rem;
        font-size: 1rem;
        height: 150px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(30, 136, 229, 0.3);
        transform: translateY(-2px);
    }
    
    .code-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #1E88E5;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    
    .code-header:before {
        content: "💻";
        margin-right: 0.5rem;
    }
    
    .output-section {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid #4CAF50;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2196F3;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #FF9800;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #F44336;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 500;
        border-radius: 10px;
        margin-right: 0.5rem;
    }
    
    .badge-blue {
        background-color: #e3f2fd;
        color: #1E88E5;
    }
    
    .badge-green {
        background-color: #e8f5e9;
        color: #4CAF50;
    }
    
    .badge-orange {
        background-color: #fff3e0;
        color: #FF9800;
    }
    
    .badge-red {
        background-color: #ffebee;
        color: #F44336;
    }
    
    .tab-content {
        padding: 1rem;
        background-color: white;
        border-radius: 0 10px 10px 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Animated loading spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0, 0, 0, 0.1);
        border-radius: 50%;
        border-top-color: #1E88E5;
        animation: spin 1s ease-in-out infinite;
        margin-right: 0.5rem;
    }
    
    .feature-icon {
        font-size: 2rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Customized progress bar */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    
    /* Customized metrics */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Chat bubbles */
    .chat-bubble {
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        max-width: 80%;
    }
    
    .user-bubble {
        background-color: #E3F2FD;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    
    .assistant-bubble {
        background-color: #F5F5F5;
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    
    /* Animation for new elements */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Media queries for responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY in your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize session state variables if they don't exist
# Replace the session state initialization section with this updated version
# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_info' not in st.session_state:
    st.session_state.df_info = ""
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'code_output' not in st.session_state:
    st.session_state.code_output = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "gpt-4o"
if 'visualization_type' not in st.session_state:
    st.session_state.visualization_type = "plotly"
if 'automated_insights' not in st.session_state:
    st.session_state.automated_insights = []
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'executed_code_history' not in st.session_state:
    st.session_state.executed_code_history = []
if 'favorite_codes' not in st.session_state:
    st.session_state.favorite_codes = []
if 'code_execution_times' not in st.session_state:
    st.session_state.code_execution_times = {}
if 'visualization_language' not in st.session_state:
    st.session_state.visualization_language = "python"  # Default to Python
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = None

# Function to generate code using OpenAI
# Function to generate code using OpenAI
def generate_code(prompt, dataframe_info, visualization_type="plotly", language="python"):
    try:
        system_message = f"""You are an expert data scientist assistant specialized in {language} programming for data analysis and visualization.
Generate {language} code based on the user's request related to their data. 
The code should be concise, efficient, and ready to run. Include only the necessary code without explanations.
Include proper visualizations and analysis as requested.

IMPORTANT: 
- The dataframe is already loaded and available as the variable 'df'. DO NOT include code to load data from files.
- If using Python, focus on using {'Plotly' if visualization_type == 'plotly' else 'Matplotlib/Seaborn'} for visualizations.
- For interactive visualizations, use Plotly with full configuration for best appearance.
- Format the code to be aesthetically pleasing and well-commented.
- Include code for handling potential errors like missing values.
- Make visualizations beautiful with colors, labels, and proper formatting.

Format your response as a code block only, without any additional text or commentary."""
        
        # Prepare the information about the dataframe
        df_message = f"Here's information about the dataframe I'm working with:\n{dataframe_info}"
        
        # Only use Python language option
        combined_prompt = f"{df_message}\n\nTask: {prompt}\n\nProvide just the Python code to accomplish this task. The dataframe is already loaded as the variable 'df'. DO NOT include code to read any files. Use {visualization_type} for creating beautiful and informative visualizations."
        
        response = client.chat.completions.create(
            model=st.session_state.model_choice,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        generated_code = response.choices[0].message.content
        
        # Extract code if it's wrapped in code blocks
        code_pattern = r"```(?:\w+)?(.*?)```"
        code_match = re.search(code_pattern, generated_code, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1).strip()
        
        # Check for file loading attempts and remove them
        lines = generated_code.split('\n')
        filtered_lines = []
        for line in lines:
            if not any(pattern in line.lower() for pattern in ['read_csv', 'read_excel', 'read_json', 'read_parquet', 'read_table', 'open(', 'fetch(']):
                filtered_lines.append(line)
        
        generated_code = '\n'.join(filtered_lines)
        
        return generated_code
    
    except Exception as e:
        return f"Error generating code: {str(e)}"
# Function to generate automated insights using OpenAI
def generate_insights(dataframe_info):
    try:
        system_message = """You are an expert data scientist specialized in generating insights from data.
Based on the dataframe information provided, generate 5-7 key insights about the data.
Each insight should be specific, clear, and actionable.
Focus on patterns, relationships, anomalies, or interesting findings that would be valuable to business stakeholders.
Format your response as a JSON array of insight objects with 'title' and 'description' fields.
Example: [{"title": "Insight 1 Title", "description": "Detailed explanation of insight 1"}, ...]"""
        
        prompt = f"Here's information about the dataframe:\n{dataframe_info}\n\nGenerate key insights from this data."
        
        response = client.chat.completions.create(
            model=st.session_state.model_choice,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
            # Remove the response_format parameter that's causing the error
            # response_format={"type": "json_object"}
        )
        
        insights_text = response.choices[0].message.content
        
        # Since we're not forcing JSON response, we need to handle potential non-JSON responses
        try:
            # Try to parse as JSON
            import json
            import re
            
            # Check if the response looks like JSON
            if insights_text.strip().startswith('[') and insights_text.strip().endswith(']'):
                insights = json.loads(insights_text)
                return insights
            
            # If not proper JSON, try to extract JSON-like content
            json_match = re.search(r'\[\s*\{.*\}\s*\]', insights_text, re.DOTALL)
            if json_match:
                try:
                    insights = json.loads(json_match.group(0))
                    return insights
                except:
                    pass
            
            # If we couldn't parse JSON, create our own structure
            # Split by numbered items
            insights = []
            current_title = "Key Insight"
            current_description = ""
            
            lines = insights_text.split('\n')
            for line in lines:
                # Check for patterns like "1." or "Insight 1:" that might indicate a new insight
                if re.match(r'^\d+\.', line) or re.match(r'^Insight \d+:', line):
                    # Save previous insight if we have one
                    if current_description:
                        insights.append({"title": current_title, "description": current_description.strip()})
                    # Start new insight
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        current_title = parts[0].strip()
                        current_description = parts[1].strip()
                    else:
                        current_title = line.strip()
                        current_description = ""
                else:
                    current_description += " " + line.strip()
            
            # Don't forget to add the last insight
            if current_description:
                insights.append({"title": current_title, "description": current_description.strip()})
            
            return insights
            
        except Exception as json_error:
            # Return a simple structure if JSON parsing fails
            return [
                {"title": "Data Overview", "description": "The model generated insights, but they couldn't be parsed as JSON. Please regenerate insights."}
            ]
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return []


# Function to detect anomalies in the data
def detect_anomalies(df):
    try:
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 1:
            return "No numeric columns available for anomaly detection."
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Use Isolation Forest for anomaly detection
        model = IsolationForest(contamination=0.05, random_state=42)
        df_anomaly = df.copy()
        df_anomaly['anomaly_score'] = model.fit_predict(scaled_data)
        df_anomaly['is_anomaly'] = df_anomaly['anomaly_score'] == -1
        
        # Get anomalies
        anomalies = df_anomaly[df_anomaly['is_anomaly'] == True]
        
        return {
            'anomalies_df': anomalies,
            'anomaly_count': anomalies.shape[0],
            'percentage': (anomalies.shape[0] / df.shape[0]) * 100,
            'columns': numeric_df.columns.tolist()
        }
    
    except Exception as e:
        return f"Error in anomaly detection: {str(e)}"

# Function to execute the generated code
def execute_code(code, df, language="python"):
    # Record execution start time
    start_time = time.time()
    
    # Create a string buffer to capture output
    buffer = io.StringIO()
    
    # Redirect stdout to the buffer
    old_stdout = sys.stdout
    sys.stdout = buffer
    
    # Create a new figure for matplotlib
    plt.figure(figsize=(10, 6))
    
    # Create a local namespace with the dataframe
    # Handle problematic column types to prevent Arrow errors
    df_safe = df.copy()
    
    # Convert Int64 types to standard int64 to prevent Arrow errors
    for col in df_safe.columns:
        # Handle Int64 and other pandas extension types
        if str(df_safe[col].dtype).startswith(('Int', 'UInt', 'Float')):
            try:
                df_safe[col] = df_safe[col].astype('float64')
            except:
                pass
        # Handle datetime types - convert to string to prevent operations that aren't supported
        elif pd.api.types.is_datetime64_any_dtype(df_safe[col]):
            try:
                df_safe[col] = df_safe[col].astype(str)
            except:
                pass
    
    # Include plotly libraries in the local namespace
    local_namespace = {
        "df": df_safe, 
        "pd": pd, 
        "np": np, 
        "plt": plt, 
        "sns": sns,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
        "ff": ff
    }
    
    # Execute the code
    try:
        exec(code, local_namespace)
        
        # Check if there's a plotly figure in the namespace
        plotly_fig = None
        matplotlib_fig = None
        
        # Look for plotly figure in the namespace
        for var_name, var_value in local_namespace.items():
            if var_name not in ["df", "pd", "np", "plt", "sns", "px", "go", "make_subplots", "ff"]:
                # Fixed: Check for proper Plotly figure types
                if isinstance(var_value, go.Figure):
                    plotly_fig = var_value
                    break
        
        # Get the matplotlib plot if one was created
        if plt.get_fignums():
            matplotlib_fig = plt.gcf()
            if matplotlib_fig.get_axes():
                plt.tight_layout()
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plt_html = f'<img src="data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}" style="width:100%">'
                # Close the figure to prevent display in the main thread
                plt.close()
            else:
                plt_html = ""
        else:
            plt_html = ""
        
        # Restore stdout
        sys.stdout = old_stdout
        text_output = buffer.getvalue()
        
        # Record execution end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store the execution time
        st.session_state.code_execution_times[len(st.session_state.executed_code_history)] = execution_time
        
        # Add to executed code history
        st.session_state.executed_code_history.append(code)
        
        # Return text output and any figures
        return text_output, plt_html, plotly_fig, execution_time
    
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = old_stdout
        return f"Error executing code: {str(e)}", "", None, time.time() - start_time
# Function to get dataframe information
def get_dataframe_info(df):
    buffer = io.StringIO()
    
    # Make a safe copy of the dataframe to prevent Arrow errors
    df_safe = df.copy()
    
    # Convert problematic types to more compatible types
    for col in df_safe.columns:
        # Handle Int64 and other pandas extension types
        if str(df_safe[col].dtype).startswith(('Int', 'UInt', 'Float')):
            try:
                df_safe[col] = df_safe[col].astype('float64')
            except:
                pass
        # Handle datetime types
        elif pd.api.types.is_datetime64_any_dtype(df_safe[col]):
            try:
                # Keep datetimes as is for display, but note they're special
                pass
            except:
                pass
    
    # Basic info
    buffer.write(f"DataFrame Shape: {df_safe.shape[0]} rows, {df_safe.shape[1]} columns\n\n")
    
    # Column information
    buffer.write("Columns:\n")
    for col in df_safe.columns:
        dtype_str = str(df_safe[col].dtype)
        # Add special handling note for datetime columns
        if pd.api.types.is_datetime64_any_dtype(df_safe[col]):
            dtype_str += " (datetime)"
        buffer.write(f"- {col} (dtype: {dtype_str})\n")
    
    # Sample data
    buffer.write("\nSample Data (first 5 rows):\n")
    buffer.write(df_safe.head().to_string())
    
    # Numeric summary
    numeric_cols = df_safe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        buffer.write("\n\nNumeric Columns Summary:\n")
        buffer.write(df_safe[numeric_cols].describe().to_string())
    
    # Categorical summary
    cat_cols = df_safe.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        buffer.write("\n\nCategorical Columns Summary:\n")
        for col in cat_cols:
            buffer.write(f"\n{col} - Unique Values: {df_safe[col].nunique()}\n")
            buffer.write(df_safe[col].value_counts().head(5).to_string())
    
    # Datetime summary if applicable
    datetime_cols = [col for col in df_safe.columns if pd.api.types.is_datetime64_any_dtype(df_safe[col])]
    if datetime_cols:
        buffer.write("\n\nDatetime Columns Summary:\n")
        for col in datetime_cols:
            buffer.write(f"\n{col}:\n")
            buffer.write(f"- Min: {df_safe[col].min()}\n")
            buffer.write(f"- Max: {df_safe[col].max()}\n")
            buffer.write(f"- Range: {df_safe[col].max() - df_safe[col].min()}\n")
    
    # Missing values
    buffer.write("\n\nMissing Values:\n")
    buffer.write(df_safe.isnull().sum().to_string())
    
    # Correlation matrix for numeric columns (first 10 only to keep it manageable)
    if len(numeric_cols) > 1:
        buffer.write("\n\nCorrelation Matrix (first 10 numeric columns):\n")
        corr_cols = numeric_cols[:10]  # Take only first 10 columns
        buffer.write(df_safe[corr_cols].corr().to_string())
    
    return buffer.getvalue()

# Function to generate suggested questions based on dataframe
def generate_suggested_questions(dataframe_info):
    try:
        system_message = """You are an expert data scientist assistant.
Based on the dataframe information provided, generate 5 suggested questions or analysis tasks that would be insightful for this dataset.
Each suggestion should be specific to the data structure and content.
Format your response as a JSON array of strings.
Example: ["What is the correlation between X and Y?", "Can you create a scatter plot of A vs B?", ...]"""
        
        prompt = f"Here's information about the dataframe:\n{dataframe_info}\n\nGenerate 5 suggested questions or analysis tasks for this data."
        
        response = client.chat.completions.create(
            model=st.session_state.model_choice,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
            # Remove response_format parameter
            # response_format={"type": "json_object"}
        )
        
        suggestions_text = response.choices[0].message.content
        
        # Handle potential non-JSON responses
        try:
            import json
            import re
            
            # Try to parse as JSON
            if suggestions_text.strip().startswith('[') and suggestions_text.strip().endswith(']'):
                suggestions = json.loads(suggestions_text)
                return suggestions
            
            # If not proper JSON, try to extract questions
            questions = []
            lines = suggestions_text.split('\n')
            for line in lines:
                # Look for numbered items or questions
                if re.match(r'^\d+\.', line) or '?' in line:
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question:
                        questions.append(question)
            
            if questions:
                return questions
            
            # Fallback: split by newlines and return non-empty lines
            return [line.strip() for line in suggestions_text.split('\n') if line.strip()]
            
        except Exception as json_error:
            # Return basic questions if parsing fails
            return ["Could not generate suggestions. Please try again."]
    
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return ["Could not generate suggestions. Please try again."]

# Function to perform automated EDA
def perform_automated_eda(df):
    # Initialize insights dictionary
    insights = {}
    
    try:
        # Basic statistics
        insights["basic_stats"] = {}
        insights["basic_stats"]["rows"] = df.shape[0]
        insights["basic_stats"]["columns"] = df.shape[1]
        insights["basic_stats"]["missing_values"] = df.isnull().sum().sum()
        insights["basic_stats"]["missing_percentage"] = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        
        # Column types
        insights["column_types"] = {}
        insights["column_types"]["numeric"] = len(df.select_dtypes(include=[np.number]).columns)
        insights["column_types"]["categorical"] = len(df.select_dtypes(include=['object', 'category']).columns)
        insights["column_types"]["datetime"] = len([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])])
        insights["column_types"]["boolean"] = len(df.select_dtypes(include=['bool']).columns)
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights["numeric_stats"] = {}
            for col in numeric_cols:
                insights["numeric_stats"][col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis()
                }
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            # Get correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Find highly correlated pairs (absolute correlation > 0.7)
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append({
                            "col1": numeric_cols[i],
                            "col2": numeric_cols[j],
                            "correlation": corr_matrix.iloc[i, j]
                        })
            
            insights["correlations"] = {
                "high_corr_pairs": high_corr_pairs
            }
        
        # Categorical column analysis
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            insights["categorical_stats"] = {}
            for col in cat_cols:
                insights["categorical_stats"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_value": df[col].value_counts().index[0] if df[col].value_counts().shape[0] > 0 else None,
                    "top_count": df[col].value_counts().iloc[0] if df[col].value_counts().shape[0] > 0 else 0,
                    "top_percentage": (df[col].value_counts().iloc[0] / df.shape[0]) * 100 if df[col].value_counts().shape[0] > 0 else 0
                }
        
        # Dimensionality reduction for visualization (if enough numeric columns)
        if len(numeric_cols) >= 2:
            # Handle missing values for PCA
            df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_numeric)
            
            # Perform PCA
            if df_numeric.shape[1] >= 3:  # Only do PCA if we have at least 3 numeric columns
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(scaled_data)
                
                insights["pca"] = {
                    "explained_variance": pca.explained_variance_ratio_.tolist(),
                    "total_explained_variance": sum(pca.explained_variance_ratio_),
                    "pca_columns": ["PC1", "PC2", "PC3"]
                }
        
        # Clustering analysis (if enough data and numeric columns)
        if len(numeric_cols) >= 2 and df.shape[0] >= 50:
            # Handle missing values
            df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_numeric)
            
            # Determine optimal number of clusters using silhouette score
            silhouette_scores = []
            max_clusters = min(10, df.shape[0] // 5)  # Max of 10 clusters or 1/5 of data size
            
            for n_clusters in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_data)
                    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
                    silhouette_scores.append({"n_clusters": n_clusters, "score": silhouette_avg})
                except:
                    continue
            
            if silhouette_scores:
                # Get best number of clusters
                best_n_clusters = max(silhouette_scores, key=lambda x: x["score"])["n_clusters"]
                
                # Perform clustering with optimal clusters
                kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
                df_clusters = df.copy()
                df_clusters['cluster'] = kmeans.fit_predict(scaled_data)
                
                # Get cluster sizes
                cluster_sizes = df_clusters['cluster'].value_counts().to_dict()
                
                insights["clustering"] = {
                    "method": "KMeans",
                    "optimal_clusters": best_n_clusters,
                    "silhouette_score": max(silhouette_scores, key=lambda x: x["score"])["score"],
                    "cluster_sizes": cluster_sizes
                }
        
        return insights
    
    except Exception as e:
        return {"error": str(e)}

# Function to load sample dataset
def load_sample_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['species'] = pd.Categorical.from_codes(data.target, data.target_names)
        return df
    
    elif dataset_name == "Titanic Dataset":
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        return pd.read_csv(url)
    
    elif dataset_name == "California Housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['PRICE'] = data.target
        return df
    
    elif dataset_name == "Diabetes Dataset":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    elif dataset_name == "Wine Dataset":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['wine_type'] = pd.Categorical.from_codes(data.target, data.target_names)
        return df
    
    elif dataset_name == "NBA Players":
        url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-players/nba_2022.csv"
        return pd.read_csv(url)
    
    elif dataset_name == "COVID-19 Data":
        url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/latest/owid-covid-latest.csv"
        return pd.read_csv(url)
    
    elif dataset_name == "E-commerce Sales":
        # Create synthetic e-commerce dataset
        np.random.seed(42)
        n = 1000
        
        # Create date range
        date_range = pd.date_range(start='2023-01-01', periods=n, freq='D')
        
        # Create product categories
        categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Books']
        
        # Create dataframe
        df = pd.DataFrame({
            'date': np.random.choice(date_range, size=n),
            'category': np.random.choice(categories, size=n),
            'product_id': np.random.randint(1000, 9999, size=n),
            'price': np.random.uniform(10, 500, size=n).round(2),
            'quantity': np.random.randint(1, 10, size=n),
            'customer_id': np.random.randint(100, 999, size=n),
            'review_score': np.random.randint(1, 6, size=n),
            'discount_applied': np.random.choice([True, False], size=n, p=[0.3, 0.7])
        })
        
        # Calculate total sales
        df['total_sales'] = df['price'] * df['quantity']
        
        # Apply discount
        df.loc[df['discount_applied'], 'total_sales'] = df.loc[df['discount_applied'], 'total_sales'] * 0.9
        
        return df
    
    return None
def modified_generate_code_tab():
    st.markdown('<div class="sub-header">Generate Code</div>', unsafe_allow_html=True)
    
    # Examples and suggestions
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### Example Queries")
    
    # Get suggested questions based on the dataframe
    if not hasattr(st.session_state, 'suggested_questions') or st.session_state.suggested_questions is None:
        with st.spinner("Generating suggested questions..."):
            st.session_state.suggested_questions = generate_suggested_questions(st.session_state.df_info)
    
    # Display suggested questions
    st.markdown("**Try asking:**")
    suggested_questions = st.session_state.suggested_questions
    
    # Create columns for suggestion buttons
    col1, col2 = st.columns(2)
    
    for i, question in enumerate(suggested_questions[:6]):  # Limit to 6 suggestions
        with col1 if i % 2 == 0 else col2:
            if st.button(f"{question}", key=f"suggestion_{i}"):
                st.session_state.prompt = question
                with st.spinner("Generating code..."):
                    st.session_state.generated_code = generate_code(
                        question, 
                        st.session_state.df_info, 
                        st.session_state.visualization_type,
                        "python"  # Always use Python
                    )
                    st.session_state.chat_history.append({"prompt": question, "code": st.session_state.generated_code})
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # User prompt for code generation
    prompt = st.text_area("What would you like to do with your data?", 
                          value=st.session_state.get('prompt', ''),
                          placeholder="Example: Create an interactive scatter plot of price vs. rating colored by category",
                          height=100)
    
    # Options for code generation - only show visualization type
    viz_type = st.radio("Visualization Type", 
                        options=["Plotly (Interactive)", "Matplotlib/Seaborn (Static)"],
                        index=0 if st.session_state.visualization_type == "plotly" else 1)
    
    st.session_state.visualization_type = "plotly" if viz_type == "Plotly (Interactive)" else "matplotlib"
    
    # Generate code button with improved styling
    if st.button("🚀 Generate Code", use_container_width=True):
        if prompt:
            with st.spinner("Generating code..."):
                st.session_state.prompt = prompt
                st.session_state.generated_code = generate_code(
                    prompt, 
                    st.session_state.df_info, 
                    st.session_state.visualization_type,
                    "python"  # Always use Python
                )
                st.session_state.chat_history.append({"prompt": prompt, "code": st.session_state.generated_code})
        else:
            st.warning("Please enter a prompt to generate code.")
# Function to generate shareable report
def generate_report(df, insights, executed_code, plots):
    try:
        # Create report content
        report = io.StringIO()
        
        # Add header
        report.write("# Automated Data Analysis Report\n\n")
        report.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Add dataset overview
        report.write("## Dataset Overview\n\n")
        report.write(f"* **Rows:** {df.shape[0]}\n")
        report.write(f"* **Columns:** {df.shape[1]}\n")
        
        # Add basic insights
        if "basic_stats" in insights:
            report.write(f"* **Missing Values:** {insights['basic_stats']['missing_values']} ({insights['basic_stats']['missing_percentage']:.2f}%)\n")
        
        report.write("\n### Column Types\n\n")
        if "column_types" in insights:
            report.write(f"* **Numeric:** {insights['column_types']['numeric']}\n")
            report.write(f"* **Categorical:** {insights['column_types']['categorical']}\n")
            report.write(f"* **Datetime:** {insights['column_types']['datetime']}\n")
            report.write(f"* **Boolean:** {insights['column_types']['boolean']}\n")
        
        # Add sample data
        report.write("\n## Sample Data\n\n")
        report.write("```\n")
        report.write(df.head().to_string())
        report.write("\n```\n\n")
        
        # Add key insights
        report.write("## Key Insights\n\n")
        
        # Numeric statistics
        if "numeric_stats" in insights:
            report.write("### Numeric Column Statistics\n\n")
            for col, stats in insights["numeric_stats"].items():
                report.write(f"#### {col}\n\n")
                report.write(f"* **Mean:** {stats['mean']:.2f}\n")
                report.write(f"* **Median:** {stats['median']:.2f}\n")
                report.write(f"* **Min:** {stats['min']:.2f}\n")
                report.write(f"* **Max:** {stats['max']:.2f}\n")
                report.write(f"* **Standard Deviation:** {stats['std']:.2f}\n\n")
        
        # Correlations
        if "correlations" in insights and "high_corr_pairs" in insights["correlations"]:
            report.write("### High Correlations\n\n")
            for pair in insights["correlations"]["high_corr_pairs"]:
                report.write(f"* **{pair['col1']}** and **{pair['col2']}**: {pair['correlation']:.2f}\n")
            report.write("\n")
        
        # Clustering
        if "clustering" in insights:
            report.write("### Clustering Analysis\n\n")
            report.write(f"* **Optimal Clusters:** {insights['clustering']['optimal_clusters']}\n")
            report.write(f"* **Silhouette Score:** {insights['clustering']['silhouette_score']:.2f}\n")
            report.write("* **Cluster Sizes:**\n")
            for cluster, size in insights['clustering']['cluster_sizes'].items():
                report.write(f"  * Cluster {cluster}: {size} samples\n")
            report.write("\n")
        
        # Add executed code
        if executed_code:
            report.write("## Analysis Code\n\n")
            report.write("```python\n")
            report.write(executed_code)
            report.write("\n```\n\n")
        
        # Return report content
        return report.getvalue()
    
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Add content to the sidebar
# Add content to the sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Add logo or header
    st.image("https://img.icons8.com/fluent/48/000000/data-configuration.png", width=80)
    st.markdown("<h3 style='text-align: center;'>Advanced Data<br>Science Assistant</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div class="sub-header">Settings</div>', unsafe_allow_html=True)
    
    # Model selection
    model_options = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    selected_model = st.selectbox("Select OpenAI Model", 
                                 options=model_options, 
                                 index=model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else 0)
    
    st.session_state.model_choice = selected_model
    
    # Visualization preference
    viz_options = ["plotly", "matplotlib"]
    selected_viz = st.selectbox("Visualization Library", 
                               options=viz_options,
                               index=viz_options.index("plotly") if st.session_state.visualization_type == "plotly" else 1)
    
    st.session_state.visualization_type = selected_viz
    
    # Set default language to Python, but hide this option from the user
    st.session_state.visualization_language = "python"
    
    # Dark mode toggle
    dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode
    
    # Apply dark mode styling (no changes here)
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
            .stApp {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            .card, .output-section, .tab-content {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            .sub-header {
                color: #e0e0e0;
                border-bottom: 2px solid #444;
            }
            .stTextInput>div>div>input, .stTextArea>div>div>textarea {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #555;
            }
            .info-box {
                background-color: #253341;
                border-left: 4px solid #1E88E5;
            }
            .warning-box {
                background-color: #3e2723;
                border-left: 4px solid #FF9800;
            }
            .success-box {
                background-color: #1b3a26;
                border-left: 4px solid #4CAF50;
            }
            .error-box {
                background-color: #3e1a17;
                border-left: 4px solid #F44336;
            }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    
    st.markdown('<div class="sub-header">Data Options</div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data file", 
                                    type=["csv", "xlsx", "json", "parquet"],
                                    help="Upload a data file to analyze")
    
    # Sample datasets
    st.markdown('<div class="info-box">Or use a sample dataset:</div>', unsafe_allow_html=True)
    sample_dataset = st.selectbox("Select Sample Dataset", 
                                 options=["None", "Iris Dataset", "Titanic Dataset", "California Housing", "Diabetes Dataset", 
                                         "Wine Dataset", "NBA Players", "COVID-19 Data", "E-commerce Sales"])
    
    if st.button("Load Sample Dataset"):
        if sample_dataset != "None":
            with st.spinner(f"Loading {sample_dataset}..."):
                st.session_state.df = load_sample_dataset(sample_dataset)
                if st.session_state.df is not None:
                    st.session_state.df_info = get_dataframe_info(st.session_state.df)
                    st.session_state.sample_data_loaded = True
                    st.success(f"Loaded {sample_dataset} successfully!")
                    
                    # Generate automated insights
                    with st.spinner("Generating automated insights..."):
                        st.session_state.automated_insights = generate_insights(st.session_state.df_info)
                    
                    # Detect anomalies
                    with st.spinner("Detecting anomalies..."):
                        st.session_state.anomalies = detect_anomalies(st.session_state.df)
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded file..."):
                if uploaded_file.name.endswith('.csv'):
                    # Try different encodings
                    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                    
                    for encoding in encodings:
                        try:
                            st.session_state.df = pd.read_csv(uploaded_file, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            # Reset file pointer for next attempt
                            uploaded_file.seek(0)
                            continue
                        except Exception as e:
                            st.error(f"Error loading CSV with {encoding} encoding: {e}")
                            break
                            
                    if st.session_state.df is None:
                        st.error("Failed to read the CSV file with any of the attempted encodings.")
                        
                elif uploaded_file.name.endswith('.xlsx'):
                    st.session_state.df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    st.session_state.df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    st.session_state.df = pd.read_parquet(uploaded_file)
                
                if st.session_state.df is not None:
                    st.session_state.df_info = get_dataframe_info(st.session_state.df)
                    st.success(f"Uploaded {uploaded_file.name} successfully!")
                    
                    # Generate automated insights
                    with st.spinner("Generating automated insights..."):
                        st.session_state.automated_insights = generate_insights(st.session_state.df_info)
                    
                    # Detect anomalies
                    with st.spinner("Detecting anomalies..."):
                        st.session_state.anomalies = detect_anomalies(st.session_state.df)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Try checking if the file is corrupted or in an unexpected format. You can also try converting it to a different format before uploading.")
    
    st.markdown('<div class="warning-box">Important: Sample data will be cleared if you upload a file.</div>', unsafe_allow_html=True)
    
    # Clear data button
    if st.button("Clear Data"):
        st.session_state.df = None
        st.session_state.df_info = ""
        st.session_state.generated_code = ""
        st.session_state.code_output = ""
        st.session_state.sample_data_loaded = False
        st.session_state.automated_insights = []
        st.session_state.anomalies = None
        st.session_state.executed_code_history = []
        st.session_state.favorite_codes = []
        st.session_state.code_execution_times = {}
        st.success("Data cleared successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main page content
st.markdown('<div class="main-header">Advanced Data Science AI Assistant</div>', unsafe_allow_html=True)

# Display data information if available
if st.session_state.df is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "AI Insights", "Generate Code", "Visualization Gallery", "Export"])
    
    with tab1:
        st.markdown('<div class="sub-header">Data Overview</div>', unsafe_allow_html=True)
        
        # Display basic dataframe info
        df_shape = st.session_state.df.shape
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df_shape[0]:,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df_shape[1]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
        
        with col3:
            missing_values = st.session_state.df.isnull().sum().sum()
            missing_percentage = (missing_values / (df_shape[0] * df_shape[1])) * 100
            st.markdown(f'<div class="metric-card"><div class="metric-value">{missing_values:,}</div><div class="metric-label">Missing Values ({missing_percentage:.1f}%)</div></div>', unsafe_allow_html=True)
        
        with col4:
            numeric_cols = len(st.session_state.df.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(st.session_state.df.select_dtypes(include=['object', 'category']).columns)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{numeric_cols} / {categorical_cols}</div><div class="metric-label">Numeric / Categorical</div></div>', unsafe_allow_html=True)
        
        # Show the dataframe with improved styling
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(st.session_state.df.head(10), height=300, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data info tabs
        info_tab1, info_tab2, info_tab3, info_tab4 = st.tabs(["Column Info", "Summary Statistics", "Missing Values", "Correlations"])
        
        with info_tab1:
            # Column information
            col_info = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Type': st.session_state.df.dtypes,
                'Unique Values': [st.session_state.df[col].nunique() for col in st.session_state.df.columns],
                'Missing Values': st.session_state.df.isnull().sum().values,
                'Missing Percentage': (st.session_state.df.isnull().sum().values / len(st.session_state.df) * 100).round(2)
            })
            st.dataframe(col_info, height=300, use_container_width=True)
        
        with info_tab2:
            # Summary statistics for numeric columns
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Columns**")
                st.dataframe(st.session_state.df[numeric_cols].describe(), use_container_width=True)
            
            # Summary for categorical columns
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                st.markdown("**Categorical Columns**")
                for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                    st.markdown(f"**{col}** - Top 5 values")
                    value_counts = st.session_state.df[col].value_counts().head(5)
                    value_df = pd.DataFrame({
                        col: value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(st.session_state.df) * 100).round(2)
                    })
                    st.dataframe(value_df, use_container_width=True)
        
        with info_tab3:
            # Missing values visualization
            st.markdown("**Missing Values Heatmap**")
            
            # Create missing values heatmap using plotly
            missing_data = st.session_state.df.isnull()
            fig = px.imshow(
                missing_data.T,
                labels=dict(x="Row Index", y="Column", color="Is Missing"),
                color_continuous_scale=["#ffffff", "#1E88E5"],
                title="Missing Values Heatmap",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Missing values table
            missing_data = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Missing Values': st.session_state.df.isnull().sum().values,
                'Percentage': (st.session_state.df.isnull().sum().values / len(st.session_state.df) * 100).round(2)
            })
            st.dataframe(missing_data.sort_values('Missing Values', ascending=False), use_container_width=True)
        
        with info_tab4:
            # Correlation heatmap for numeric columns
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.markdown("**Correlation Heatmap**")
                corr = st.session_state.df[numeric_cols].corr()
                
                # Create plotly correlation heatmap
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    x=corr.columns,
                    y=corr.columns,
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show highest correlations
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_pairs.append({
                            'Feature 1': numeric_cols[i],
                            'Feature 2': numeric_cols[j],
                            'Correlation': corr.iloc[i, j]
                        })
                
                if corr_pairs:
                    # Sort by absolute correlation
                    corr_pairs_df = pd.DataFrame(corr_pairs)
                    corr_pairs_df['Abs Correlation'] = corr_pairs_df['Correlation'].abs()
                    corr_pairs_df = corr_pairs_df.sort_values('Abs Correlation', ascending=False).drop('Abs Correlation', axis=1)
                    
                    st.markdown("**Highest Feature Correlations**")
                    st.dataframe(corr_pairs_df.head(10), use_container_width=True)
            else:
                st.info("Not enough numeric columns to calculate correlations.")
    
    with tab2:
        st.markdown('<div class="sub-header">AI Insights</div>', unsafe_allow_html=True)
        
        insight_tab1, insight_tab2, insight_tab3 = st.tabs(["Key Insights", "Anomaly Detection", "Auto EDA"])
        
        with insight_tab1:
            st.markdown("## Key Insights from Your Data")
            
            # Display automated insights
            if st.session_state.automated_insights:
                for i, insight in enumerate(st.session_state.automated_insights):
                    st.markdown(f'<div class="card"><h3>{insight["title"]}</h3><p>{insight["description"]}</p></div>', unsafe_allow_html=True)
            else:
                st.info("Generating insights... This may take a moment.")
                with st.spinner("Generating insights..."):
                    st.session_state.automated_insights = generate_insights(st.session_state.df_info)
                st.experimental_rerun()
            
            # Refresh insights button
            if st.button("Refresh Insights"):
                with st.spinner("Refreshing insights..."):
                    st.session_state.automated_insights = generate_insights(st.session_state.df_info)
                st.success("Insights refreshed!")
        
        with insight_tab2:
            st.markdown("## Anomaly Detection")
            
            # Display anomaly detection results
            if st.session_state.anomalies:
                if isinstance(st.session_state.anomalies, dict) and 'anomalies_df' in st.session_state.anomalies:
                    anomaly_df = st.session_state.anomalies['anomalies_df']
                    
                    # Display anomaly metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.anomalies["anomaly_count"]}</div><div class="metric-label">Anomalies</div></div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.anomalies["percentage"]:.1f}%</div><div class="metric-label">Anomaly Percentage</div></div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.anomalies["columns"])}</div><div class="metric-label">Columns Analyzed</div></div>', unsafe_allow_html=True)
                    
                    # Display anomalies table
                    st.markdown("### Detected Anomalies")
                    st.dataframe(anomaly_df.drop(['anomaly_score', 'is_anomaly'], axis=1), use_container_width=True)
                    
                    # Create visualization of anomalies
                    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) >= 2:
                        # Choose first two numeric columns for visualization
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                        
                        st.markdown("### Anomaly Visualization")
                        
                        # Create scatter plot with anomalies highlighted
                        fig = px.scatter(
                            anomaly_df,
                            x=x_col,
                            y=y_col,
                            color='is_anomaly',
                            color_discrete_map={True: '#FF5252', False: '#4CAF50'},
                            title=f"Anomalies in {x_col} vs {y_col}",
                            labels={'is_anomaly': 'Is Anomaly'},
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(str(st.session_state.anomalies))
            else:
                st.info("Detecting anomalies... This may take a moment.")
                with st.spinner("Detecting anomalies..."):
                    st.session_state.anomalies = detect_anomalies(st.session_state.df)
                st.experimental_rerun()
            
            # Refresh anomalies button
            if st.button("Refresh Anomaly Detection"):
                with st.spinner("Refreshing anomaly detection..."):
                    st.session_state.anomalies = detect_anomalies(st.session_state.df)
                st.success("Anomaly detection refreshed!")
        
        with insight_tab3:
            st.markdown("## Automated Exploratory Data Analysis")
            
            # Perform automated EDA
            with st.spinner("Performing automated EDA..."):
                eda_insights = perform_automated_eda(st.session_state.df)
            
            # Display EDA insights
            if isinstance(eda_insights, dict) and "error" not in eda_insights:
                # Display basic statistics
                st.markdown("### Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_insights["basic_stats"]["rows"]:,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_insights["basic_stats"]["columns"]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_insights["basic_stats"]["missing_values"]:,}</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_insights["basic_stats"]["missing_percentage"]:.1f}%</div><div class="metric-label">Missing %</div></div>', unsafe_allow_html=True)
                
                # Display column type distribution
                st.markdown("### Column Type Distribution")
                
                column_types = [
                    {
                        "type": "Numeric",
                        "count": eda_insights["column_types"]["numeric"]
                    },
                    {
                        "type": "Categorical",
                        "count": eda_insights["column_types"]["categorical"]
                    },
                    {
                        "type": "Datetime",
                        "count": eda_insights["column_types"]["datetime"]
                    },
                    {
                        "type": "Boolean",
                        "count": eda_insights["column_types"]["boolean"]
                    }
                ]
                
                fig = px.pie(
                    column_types, 
                    names="type", 
                    values="count",
                    title="Column Type Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display correlations if available
                if "correlations" in eda_insights and "high_corr_pairs" in eda_insights["correlations"]:
                    st.markdown("### High Correlations")
                    
                    high_corr_pairs = eda_insights["correlations"]["high_corr_pairs"]
                    
                    if high_corr_pairs:
                        # Create dataframe from high correlation pairs
                        high_corr_df = pd.DataFrame(high_corr_pairs)
                        
                        # Create horizontal bar chart of correlations
                        high_corr_df["pair"] = high_corr_df.apply(lambda x: f"{x['col1']} / {x['col2']}", axis=1)
                        high_corr_df["abs_correlation"] = high_corr_df["correlation"].abs()
                        
                        # Sort by absolute correlation
                        high_corr_df = high_corr_df.sort_values("abs_correlation", ascending=True)
                        
                        # Create color map based on correlation direction
                        colors = high_corr_df["correlation"].apply(lambda x: "#4CAF50" if x > 0 else "#F44336")
                        
                        fig = px.bar(
                            high_corr_df, 
                            x="abs_correlation", 
                            y="pair",
                            title="Highest Feature Correlations",
                            orientation="h",
                            color="correlation",
                            color_continuous_scale="RdBu_r",
                            labels={"abs_correlation": "Absolute Correlation", "pair": "Feature Pair"}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No high correlations found.")
                
                # Display clustering information if available
                if "clustering" in eda_insights:
                    st.markdown("### Clustering Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_insights["clustering"]["optimal_clusters"]}</div><div class="metric-label">Optimal Clusters</div></div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_insights["clustering"]["silhouette_score"]:.2f}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)
                    
                    # Create pie chart of cluster sizes
                    cluster_sizes = eda_insights["clustering"]["cluster_sizes"]
                    
                    cluster_data = [
                        {
                            "cluster": f"Cluster {k}",
                            "size": v
                        }
                        for k, v in cluster_sizes.items()
                    ]
                    
                    fig = px.pie(
                        cluster_data, 
                        names="cluster", 
                        values="size",
                        title="Cluster Size Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.G10
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display PCA results if available
                if "pca" in eda_insights:
                    st.markdown("### Principal Component Analysis")
                    
                    # Display explained variance as a bar chart
                    explained_variance = eda_insights["pca"]["explained_variance"]
                    pca_columns = eda_insights["pca"]["pca_columns"]
                    
                    pca_data = pd.DataFrame({
                        "component": pca_columns,
                        "explained_variance": explained_variance
                    })
                    
                    fig = px.bar(
                        pca_data,
                        x="component",
                        y="explained_variance",
                        title="PCA Explained Variance",
                        labels={"explained_variance": "Explained Variance Ratio", "component": "Principal Component"}
                    )
                    
                    fig.update_layout(yaxis_tickformat=".0%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"**Total Explained Variance**: {eda_insights['pca']['total_explained_variance']:.2%}")
            else:
                if "error" in eda_insights:
                    st.error(f"Error performing automated EDA: {eda_insights['error']}")
                else:
                    st.error("Error performing automated EDA.")
    
    with tab3:
        modified_generate_code_tab()
def modified_generate_code_tab():
    st.markdown('<div class="sub-header">Generate Code</div>', unsafe_allow_html=True)
    
    # Examples and suggestions
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### Example Queries")
    
    # Get suggested questions based on the dataframe
    if not hasattr(st.session_state, 'suggested_questions') or st.session_state.suggested_questions is None:
        with st.spinner("Generating suggested questions..."):
            st.session_state.suggested_questions = generate_suggested_questions(st.session_state.df_info)
    
    # Display suggested questions
    st.markdown("**Try asking:**")
    suggested_questions = st.session_state.suggested_questions
    
    # Create columns for suggestion buttons
    col1, col2 = st.columns(2)
    
    for i, question in enumerate(suggested_questions[:6]):  # Limit to 6 suggestions
        with col1 if i % 2 == 0 else col2:
            if st.button(f"{question}", key=f"suggestion_{i}"):
                st.session_state.prompt = question
                with st.spinner("Generating code..."):
                    st.session_state.generated_code = generate_code(
                        question, 
                        st.session_state.df_info, 
                        st.session_state.visualization_type,
                        "python"  # Always use Python
                    )
                    st.session_state.chat_history.append({"prompt": question, "code": st.session_state.generated_code})
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # User prompt for code generation
    prompt = st.text_area("What would you like to do with your data?", 
                          value=st.session_state.get('prompt', ''),
                          placeholder="Example: Create an interactive scatter plot of price vs. rating colored by category",
                          height=100)
    
    # Options for code generation - only show visualization type
    viz_type = st.radio("Visualization Type", 
                        options=["Plotly (Interactive)", "Matplotlib/Seaborn (Static)"],
                        index=0 if st.session_state.visualization_type == "plotly" else 1)
    
    st.session_state.visualization_type = "plotly" if viz_type == "Plotly (Interactive)" else "matplotlib"
    
    # Generate code button with improved styling
    if st.button("🚀 Generate Code", use_container_width=True):
        if prompt:
            with st.spinner("Generating code..."):
                st.session_state.prompt = prompt
                st.session_state.generated_code = generate_code(
                    prompt, 
                    st.session_state.df_info, 
                    st.session_state.visualization_type,
                    "python"  # Always use Python
                )
                st.session_state.chat_history.append({"prompt": prompt, "code": st.session_state.generated_code})
        else:
            st.warning("Please enter a prompt to generate code.")        
# This code block should replace the section where the generated code is displayed
# and the copy button is created (after the modified_generate_code_tab function)

if st.session_state.generated_code:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="code-header">Generated Code:</div>', unsafe_allow_html=True)
    st.code(st.session_state.generated_code, language=st.session_state.visualization_language)
    
    # Create a unique key for the button to avoid conflicts
    copy_button_key = f"copy_button_{hash(st.session_state.generated_code)}"
    
    # Add copy to clipboard button using Streamlit components
    if st.button("📋 Copy Code", key=copy_button_key):
        # Properly escape the code for JavaScript
        escaped_code = st.session_state.generated_code.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('\n', '\\n')
        
        # Use st.write with unsafe_allow_html to inject JavaScript
        st.write(
            f"""
            <script>
            navigator.clipboard.writeText("{escaped_code}");
            </script>
            <div class="success-box">Code copied to clipboard!</div>
            """, 
            unsafe_allow_html=True
        )
    
    # Execute code button with improved styling
    if st.button("▶️ Execute Code", use_container_width=True):
        with st.spinner("Executing code..."):
            output_text, output_plot, plotly_fig, execution_time = execute_code(
                st.session_state.generated_code, 
                st.session_state.df,
                st.session_state.visualization_language
            )
            
            # Display output
            st.markdown('<div class="output-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="code-header">Output (executed in {execution_time:.2f} seconds):</div>', unsafe_allow_html=True)
            
            # Display text output
            if output_text:
                st.text(output_text)
            
            # Display plotly figure if available
            if plotly_fig:
                st.plotly_chart(plotly_fig, use_container_width=True)
            
            # Display matplotlib plot if available
            if output_plot:
                st.markdown(output_plot, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add to favorites option
            if st.button("⭐ Add to Favorites"):
                if st.session_state.generated_code not in [item["code"] for item in st.session_state.favorite_codes]:
                    st.session_state.favorite_codes.append({
                        "prompt": st.session_state.prompt,
                        "code": st.session_state.generated_code,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success("Added to favorites!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="sub-header">Visualization Gallery</div>', unsafe_allow_html=True)
        
        # Chat history tab
        history_tab, favorites_tab = st.tabs(["Chat History", "Favorites"])
        
        with history_tab:
            if st.session_state.chat_history:
                for i, item in enumerate(st.session_state.chat_history):
                    with st.expander(f"Request {i+1}: {item['prompt'][:50]}{'...' if len(item['prompt']) > 50 else ''}"):
                        st.markdown(f"**Prompt**: {item['prompt']}")
                        st.code(item['code'], language=st.session_state.visualization_language)
                        
                        # Execute button for history items
                        if st.button(f"▶️ Execute", key=f"execute_history_{i}"):
                            with st.spinner("Executing code..."):
                                output_text, output_plot, plotly_fig, execution_time = execute_code(
                                    item['code'], 
                                    st.session_state.df,
                                    st.session_state.visualization_language
                                )
                                
                                # Display output
                                st.markdown('<div class="output-section">', unsafe_allow_html=True)
                                st.markdown(f'<div class="code-header">Output (executed in {execution_time:.2f} seconds):</div>', unsafe_allow_html=True)
                                
                                # Display text output
                                if output_text:
                                    st.text(output_text)
                                
                                # Display plotly figure if available
                                if plotly_fig:
                                    st.plotly_chart(plotly_fig, use_container_width=True)
                                
                                # Display matplotlib plot if available
                                if output_plot:
                                    st.markdown(output_plot, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No chat history yet. Generate some code to see it here!")
        
        with favorites_tab:
            if st.session_state.favorite_codes:
                for i, item in enumerate(st.session_state.favorite_codes):
                    with st.expander(f"Favorite {i+1}: {item['prompt'][:50]}{'...' if len(item['prompt']) > 50 else ''}"):
                        st.markdown(f"**Prompt**: {item['prompt']}")
                        st.markdown(f"**Added on**: {item['timestamp']}")
                        st.code(item['code'], language=st.session_state.visualization_language)
                        
                        # Execute button for favorites
                        if st.button(f"▶️ Execute", key=f"execute_favorite_{i}"):
                            with st.spinner("Executing code..."):
                                output_text, output_plot, plotly_fig, execution_time = execute_code(
                                    item['code'], 
                                    st.session_state.df,
                                    st.session_state.visualization_language
                                )
                                
                                # Display output
                                st.markdown('<div class="output-section">', unsafe_allow_html=True)
                                st.markdown(f'<div class="code-header">Output (executed in {execution_time:.2f} seconds):</div>', unsafe_allow_html=True)
                                
                                # Display text output
                                if output_text:
                                    st.text(output_text)
                                
                                # Display plotly figure if available
                                if plotly_fig:
                                    st.plotly_chart(plotly_fig, use_container_width=True)
                                
                                # Display matplotlib plot if available
                                if output_plot:
                                    st.markdown(output_plot, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Remove from favorites
                        if st.button(f"🗑️ Remove", key=f"remove_favorite_{i}"):
                            st.session_state.favorite_codes.pop(i)
                            st.success("Removed from favorites!")
                            st.experimental_rerun()
            else:
                st.info("No favorites yet. Add code to favorites to see it here!")
    
    with tab5:
        st.markdown('<div class="sub-header">Export & Share</div>', unsafe_allow_html=True)
        
        # Export options
        export_tab1, export_tab2 = st.tabs(["Export Report", "Download Data"])
        
        with export_tab1:
            st.markdown("### Export Analysis Report")
            
            st.markdown("""
            Generate a comprehensive report of your data analysis that you can download and share with others.
            The report includes:
            - Dataset overview
            - Key statistics and insights
            - Visualizations
            - Generated code
            """)
            
            # Report format selector
            report_format = st.radio("Report Format", ["Markdown", "HTML", "PDF"], horizontal=True)
            
            # Include options
            st.markdown("### Include in Report")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                include_overview = st.checkbox("Dataset Overview", value=True)
            
            with col2:
                include_insights = st.checkbox("AI Insights", value=True)
            
            with col3:
                include_code = st.checkbox("Generated Code", value=True)
            
            # Generate report button
            if st.button("📊 Generate Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    # Get last executed code or use an empty string
                    last_code = st.session_state.executed_code_history[-1] if st.session_state.executed_code_history else ""
                    
                    # Generate report markdown
                    report_content = generate_report(
                        st.session_state.df,
                        perform_automated_eda(st.session_state.df),
                        last_code if include_code else "",
                        None  # We can't easily pass plots here
                    )
                    
                    # Display report based on selected format
                    if report_format == "Markdown":
                        st.markdown(report_content)
                        
                        # Download button for Markdown
                        st.download_button(
                            label="📥 Download Markdown Report",
                            data=report_content,
                            file_name="data_analysis_report.md",
                            mime="text/markdown"
                        )
                    
                    elif report_format == "HTML":
                        # Convert markdown to HTML (simple conversion)
                        import markdown
                        html_content = f"""
                        <html>
                        <head>
                            <title>Data Analysis Report</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                                h1 {{ color: #1E88E5; }}
                                h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                                h3 {{ color: #555; }}
                                code {{ background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
                                pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                                table {{ border-collapse: collapse; width: 100%; }}
                                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                th {{ background-color: #f5f5f5; }}
                                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                            </style>
                        </head>
                        <body>
                            {markdown.markdown(report_content)}
                        </body>
                        </html>
                        """
                        
                        # Display HTML preview
                        st.markdown(html_content, unsafe_allow_html=True)
                        
                        # Download button for HTML
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=html_content,
                            file_name="data_analysis_report.html",
                            mime="text/html"
                        )
                    
                    elif report_format == "PDF":
                        st.info("PDF export functionality is coming soon! Please use Markdown or HTML formats for now.")
        
        with export_tab2:
            st.markdown("### Download Processed Data")
            
            st.markdown("""
            Download your data in various formats, including any transformations or calculations you've applied.
            """)
            
            # Export format selector
            export_format = st.radio("Export Format", ["CSV", "Excel", "JSON", "Parquet"], horizontal=True)
            
            # File name input
            file_name = st.text_input("File Name", value="processed_data")
            
            # Generate download link
            if st.button("💾 Prepare Download", use_container_width=True):
                try:
                    with st.spinner("Preparing file for download..."):
                        # Create a safe copy of the dataframe to prevent serialization issues
                        df_safe = st.session_state.df.copy()
                        
                        # Convert problematic types (like Period) to string to ensure serializability
                        for col in df_safe.columns:
                            # Handle Period objects
                            if hasattr(df_safe[col], 'dt') and hasattr(df_safe[col].dt, 'to_timestamp'):
                                try:
                                    df_safe[col] = df_safe[col].dt.to_timestamp()
                                except:
                                    df_safe[col] = df_safe[col].astype(str)
                            elif pd.api.types.is_period_dtype(df_safe[col]):
                                try:
                                    df_safe[col] = df_safe[col].dt.to_timestamp()
                                except:
                                    df_safe[col] = df_safe[col].astype(str)
                            # Handle datetime objects
                            elif pd.api.types.is_datetime64_any_dtype(df_safe[col]):
                                pass  # Keep datetime as is for most exports, handled separately for JSON
                            # Handle other potentially problematic types
                            elif len(df_safe) > 0 and isinstance(df_safe[col].iloc[0], (pd.Interval, pd.arrays.IntervalArray)):
                                df_safe[col] = df_safe[col].astype(str)
                            # Handle extension arrays like Int64, etc.
                            elif str(df_safe[col].dtype).startswith(('Int', 'UInt', 'Float')):
                                df_safe[col] = df_safe[col].astype('float64')
                        
                        if export_format == "CSV":
                            # Convert dataframe to CSV
                            csv = df_safe.to_csv(index=False)
                            
                            # Create download button
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"{file_name}.csv",
                                mime="text/csv"
                            )
                        
                        elif export_format == "Excel":
                            # Convert dataframe to Excel
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                df_safe.to_excel(writer, sheet_name='Data', index=False)
                            
                            # Create download button
                            st.download_button(
                                label="📥 Download Excel",
                                data=buffer.getvalue(),
                                file_name=f"{file_name}.xlsx",
                                mime="application/vnd.ms-excel"
                            )
                        
                        elif export_format == "JSON":
                            # For JSON specifically, convert datetimes to strings
                            df_json = df_safe.copy()
                            for col in df_json.columns:
                                if pd.api.types.is_datetime64_any_dtype(df_json[col]):
                                    df_json[col] = df_json[col].astype(str)
                            
                            # Convert dataframe to JSON
                            json_data = df_json.to_json(orient="records")
                            
                            # Create download button
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_data,
                                file_name=f"{file_name}.json",
                                mime="application/json"
                            )
                        
                        elif export_format == "Parquet":
                            # Convert dataframe to Parquet
                            buffer = io.BytesIO()
                            df_safe.to_parquet(buffer, index=False, engine='pyarrow')
                            
                            # Create download button
                            st.download_button(
                                label="📥 Download Parquet",
                                data=buffer.getvalue(),
                                file_name=f"{file_name}.parquet",
                                mime="application/octet-stream"
                            )
                
                except Exception as e:
                    st.error(f"Error preparing download: {e}")
                    # Show more detailed error info for debugging
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown(f"**Error details:** {str(e)}")
                    
                    # Identify the problematic columns if possible
                    try:
                        for col in df_safe.columns:
                            try:
                                # Try serializing each column individually
                                if export_format == "JSON":
                                    test = pd.Series(df_safe[col]).to_json()
                                elif export_format == "Parquet":
                                    test_buffer = io.BytesIO()
                                    pd.DataFrame({col: df_safe[col]}).to_parquet(test_buffer, index=False)
                            except Exception as col_error:
                                st.markdown(f"- Column '{col}' (dtype: {df_safe[col].dtype}) may be causing issues: {str(col_error)}")
                    except:
                        pass
                    
                    st.markdown("</div>", unsafe_allow_html=True)
else:
    # If no data is loaded, show welcome message
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## 👋 Welcome to the Advanced Data Science AI Assistant!
    
    This app helps you analyze your data and generate visualizations using AI-powered insights.
    
    ### 🚀 Key Features:
    
    - **Interactive Visualizations**: Create beautiful, interactive charts and plots with Plotly
    - **AI-Powered Insights**: Automatically generate insights and detect anomalies in your data
    - **Code Generation**: Turn natural language requests into Python or R code
    - **Multiple Visualization Libraries**: Choose between Plotly, Matplotlib, or Seaborn
    - **Automated EDA**: Get instant exploratory data analysis with a single click
    - **Save & Share**: Export your analyses as reports and visualizations
    
    ### 🏁 Get Started:
    
    1. Upload your own data file or select a sample dataset from the sidebar
    2. Explore the data overview and AI-generated insights
    3. Ask questions in natural language to generate visualizations and analysis
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display feature showcase
    st.markdown('<div class="sub-header">Feature Showcase</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2.5rem; text-align: center;">📊</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center;">Interactive Visualizations</h3>', unsafe_allow_html=True)
        st.markdown('Create beautiful, interactive charts and plots with Plotly. Zoom, pan, and explore your data visually.', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2.5rem; text-align: center;">🤖</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center;">AI-Powered Insights</h3>', unsafe_allow_html=True)
        st.markdown('Let AI analyze your data to discover key insights, patterns, and anomalies automatically.', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2.5rem; text-align: center;">💻</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center;">Code Generation</h3>', unsafe_allow_html=True)
        st.markdown('Turn natural language requests into Python or R code for data analysis and visualization.', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample visualizations
    st.markdown('<div class="sub-header">Sample Visualizations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://plotly.com/~tarzzz/421.png", caption="Interactive Plotly Visualization")
    
    with col2:
        st.image("https://seaborn.pydata.org/_images/seaborn-heatmap-2.png", caption="Correlation Analysis")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Advanced Data Science AI Assistant • Powered by OpenAI and Streamlit • 2025
</div>
""", unsafe_allow_html=True)