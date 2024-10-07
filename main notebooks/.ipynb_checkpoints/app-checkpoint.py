import streamlit as st
import pandas as pd
import sys
import numpy as np
import os
#from main_selected_category import processing

def main(history_data_filename, requested_formulas_data_filename, reference_formulas_data_filename, data_date, starting_date_grid_search, category_name_for_export_files = "pigment", raw_material_categories = ["PIGMENT","PATE PIG"]):
    """
    Function providing the Smart Batch results from an Excel file.
    
    Args:
        input_file (UploadedFile): Excel file provided by the user.
        
    Returns:
        output_file : Excel file containing results.
    """
    #return processing(excel_file, data_date, starting_date_grid_search, category_name_for_export_files = "pigment", raw_material_categories = ["PIGMENT","PATE PIG"])

# Interface Streamlit
st.title("Smart Batch Processing")

st.write("")
st.write("")

st.write("**Please provide the following files:**") 

history_data_obj = st.file_uploader("Batches History Data", type=["xlsx", "xls"])
requested_formulas_data_obj = st.file_uploader("Requested Formulas Data", type=["xlsx", "xls"])
reference_formulas_data_obj = st.file_uploader("Reference Formulas Data", type=["xlsx", "xls"])

if history_data_obj is not None:
    try:
        history_data_file = pd.read_excel(history_data_obj)
        st.write("Data of the file:")
        st.dataframe(df)
    except Exception as e:
        st.write("Cannot upload file:", e)

if requested_formulas_data_obj is not None:
    try:
        requested_formulas_data_file = pd.read_excel(requested_formulas_data_obj)
        st.write("Data of the file:")
        st.dataframe(df)
    except Exception as e:
        st.write("Cannot upload file:", e)

if reference_formulas_data_obj is not None:
    try:
        reference_formulas_data_file = pd.read_excel(reference_formulas_data_obj)
        st.write("Data of the file:")
        st.dataframe(df)
    except Exception as e:
        st.write("Cannot upload file:", e)

# These files should be placed in a folder named with the data date, e.g., `data 19-08-2024`. // unecessary

st.write("")
st.write("")
# Input Section with Format Guidance
st.write("**Please provide the following information:**") 

data_date = st.text_input("Data Date (DD-MM-YYYY):", placeholder="E.g., 31-12-2023")
limit_initialization_date = st.text_input("Starting Date for Grid Search (YYYY-MM-DD):", placeholder="E.g., 2020-01-01")
category_name = st.text_input("Category Name for Export Files:", placeholder="E.g., pigment") 

# Raw Material Categories (Handle as a list)
raw_materials_input = st.text_area("Raw Material Categories (comma-separated):", placeholder="E.g., ['PIGMENT','PATE PIG']")

selected_mp_categories = [category.strip() for category in raw_materials_input.split(",")]

# --- Process and Use the Inputs ---
# You can now access and use the values entered by the user
# For example:
print("Data Date:", data_date)
print("Starting Date:", limit_initialization_date)
print("Category Name:", category_name)
print("Raw Material Categories:", selected_mp_categories)



if st.button("Return results"):
    if fichier_excel is not None:
        output = main(excel_file, data_date, starting_date_grid_search, category_name_for_export_files = "pigment", raw_material_categories = ["PIGMENT","PATE PIG"])
        

    
    else:
        st.warning("Please, provide an Excel file.")
