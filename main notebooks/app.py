import streamlit as st
import pandas as pd
import sys
import numpy as np
import os
from main_selected_category import processing
from datetime import date, timedelta
from io import BytesIO
def main(history_data_file, requested_formulas_data_file, reference_formulas_data_file, data_date='06-09-2024', starting_date_grid_search='2024-03-06', category_name_for_export_files = "pigment", raw_material_categories = ["PIGMENT","PATE PIG"]):
    """
    Function providing the Smart Batch results from an Excel file.
    
    Args:
        input_file (UploadedFile): Excel file provided by the user.
        
    Returns:
        output_file : Excel file containing results.
    """
    output = processing(history_data_file, requested_formulas_data_file, reference_formulas_data_file, data_date, starting_date_grid_search, category_name_for_export_files, raw_material_categories)
    return output
    #return processing(excel_file, data_date, starting_date_grid_search, category_name_for_export_files = "pigment", raw_material_categories = ["PIGMENT","PATE PIG"])

# Interface Streamlit
st.set_page_config(
    page_title="Smart Batch Processing",
)
st.title("Smart Batch Processing")

st.write("")
st.write("")

st.write("**Please submit the requested files.**") 
#st.write("These files should be within a folder labeled 'data YY-MM-DD' (using today's date) and placed inside the 'data' folder.")
st.write("The files should follow the location: data/data DD-MM-YYYY (using today's date).")

history_data_obj = st.file_uploader("Batches History Data: `IA BPC Teinte v2.xlsm`", type=["xlsx", "xls", "xlsm"])
requested_formulas_data_obj = st.file_uploader("Requested Formulas Data: `IA BPC Teinte Besoins.xlsm`", type=["xlsx", "xls", "xlsm"])
reference_formulas_data_obj = st.file_uploader("Reference Formulas Data: `composition des SOP avec type d'intro MP.xlsm`", type=["xlsx", "xls", "xlsm"])

if history_data_obj is not None:
    try:
        history_data_file = pd.read_excel(history_data_obj)
        st.write("View of Batches History:")
        st.dataframe(history_data_file)
    except Exception as e:
        st.write("Cannot upload file:", e)

if requested_formulas_data_obj is not None:
    try:
        requested_formulas_data_file = pd.read_excel(requested_formulas_data_obj)
        st.write("View of Requested Formulas:")
        st.dataframe(requested_formulas_data_file)
    except Exception as e:
        st.write("Cannot upload file:", e)

if reference_formulas_data_obj is not None:
    try:
        reference_formulas_data_file = pd.read_excel(reference_formulas_data_obj)
        st.write("View of Reference Formulas")
        st.dataframe(reference_formulas_data_file)
    except Exception as e:
        st.write("Cannot upload file:", e)

# These files should be placed in a folder named with the data date, e.g., `data 19-08-2024`. // unecessary

st.write("")
st.write("")
# Input Section with Format Guidance
st.write("**Please provide the following information:**") 

# Calculer la date d'aujourd'hui et la date d'il y a 6 mois
today = date.today()
six_months_ago = today - timedelta(days=180)

# Formater les dates pour les champs de saisie
today_formatted = today.strftime("%d-%m-%Y")
six_months_ago_formatted = six_months_ago.strftime("%Y-%m-%d")

data_date = st.text_input("Data Date (DD-MM-YYYY):", placeholder="E.g., '06-09-2024'", value=today_formatted)
starting_date_grid_search = st.text_input("Starting Date for Grid Search (YYYY-MM-DD):", placeholder="E.g., '2024-03-06'", value=six_months_ago_formatted)
category_name_for_export_files = st.text_input("Category Name for Export Files:", placeholder="E.g., 'pigment'", value = 'pigment') 

# Raw Material Categories (Handle as a list)
raw_material_categories = st.text_area("Raw Material Categories (comma-separated):", placeholder="E.g., 'PIGMENT,PATE PIG'", value='PIGMENT,PATE PIG').split(',')

selected_mp_categories = [category.strip() for category in raw_material_categories]

# --- Process and Use the Inputs ---
# You can now access and use the values entered by the user
# For example:
print("Data Date:", data_date)
print("Starting Date:", starting_date_grid_search)
print("Category Name:", category_name_for_export_files)
print("Raw Material Categories:", raw_material_categories)

if st.button("Return results"):
    output = processing(history_data_file, requested_formulas_data_file, reference_formulas_data_file, data_date, starting_date_grid_search, category_name_for_export_files, raw_material_categories)

    # Créer un fichier Excel pour le téléchargement
    output = BytesIO()
    #with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    #    output_df.to_excel(writer, sheet_name='Résultats', index=False)

    # Bouton de téléchargement
    st.download_button(
        label="Télécharger les résultats",
        data=output.getvalue(),
        file_name=f"resultats_{category_name_for_export_files}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
#output = processing(history_data_file, requested_formulas_data_file, reference_formulas_data_file, data_date, starting_date_grid_search, category_name_for_export_files, raw_material_categories)
#st.download_button(
#        label="Download Results",
#        data=output,
#        file_name=f"results_{category_name_for_export_files}_{data_date}.xlsx",
#        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#    )