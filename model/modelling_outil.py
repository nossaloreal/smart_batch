import pandas as pd
import numpy as np
from copy import copy

SELECTED_COLUMNS = ['WipOrderNo', 'N° formule', 'Description', 'LotNo', 'UsageType',
                            'ComponentProductNo', 'Nom_MP', 'Type MP', 'ComponentLotNo',
                            'Intro Pesée en Gr', 'Rectif ou Ajust Pesée en Gr', 'Type de pesée', 'Heure début',
                            'Heure fin', 'Emballage', 'compteur', 'Skid', 'Qté OF lancée',
                            'Qté Produite', 'Ligne Annulée', 'Date Intro', 'Indice', 'NSQ',
                            'Lot Fournisseur', 'Opération', 'DU', 'Commentaire DU', 'Techno',
                            '367_GAMMES', 'Besoins 7 jours', 'Besoins 1 mois' , 'Type']

def cleaning_data(factory_df):
    """
    Filter out specific columns and transform the factory DataFrame.
    :param factory_df: A DataFrame representing the factory data.
    :type factory_df: pd.DataFrame
    """
    selected_columns = SELECTED_COLUMNS

    factory_df_cleaned = factory_df[selected_columns]

    # Filtering the dataframe for rows having the 'NSQ' column == 'NSQ3'
    factory_df_cleaned = factory_df_cleaned[(factory_df_cleaned['NSQ'] == 'NSQ3') & (factory_df_cleaned['UsageType'] != 'Introduction directe')]

    # Fill null values of column 'Type MP'
    factory_df_cleaned['Type MP'] = factory_df_cleaned['Type MP'].fillna('undefined')

    # Split the datetime column into date and time columns
    factory_df_cleaned['starting_date'] = factory_df_cleaned['Heure début'].dt.date ###date de début plutot non ?
    factory_df_cleaned['starting_time'] = factory_df_cleaned['Heure début'].dt.time

    factory_df_cleaned['ending_date'] = factory_df_cleaned['Heure fin'].dt.date ###date de fin plutot non ?
    factory_df_cleaned['ending_time'] = factory_df_cleaned['Heure fin'].dt.time
    factory_df_cleaned = factory_df_cleaned.drop(['Heure début', 'Heure fin'], axis=1) ### problème

    ### peut etre ajouter des blocs try / except pour s'assurer de la cohérence

    # Group by 'WipOrderNo' to find the earliest date
    order_creation_date = factory_df_cleaned.groupby('WipOrderNo')['starting_date'].min().reset_index()
    order_creation_date.columns = ['WipOrderNo', 'creation_date']
    order_creation_date['creation_date'] = pd.to_datetime(order_creation_date['creation_date']).dt.to_period('D')

    # Merge with the original dataframe
    factory_df_cleaned = pd.merge(factory_df_cleaned, order_creation_date, on='WipOrderNo')

    # Find orders adjusted list
    orders_adjusted_list = factory_df_cleaned[factory_df_cleaned['Rectif ou Ajust Pesée en Gr'] > 0.0]['WipOrderNo'].unique().tolist()
    factory_df_cleaned['adjusted'] = factory_df_cleaned['WipOrderNo'].isin(orders_adjusted_list).astype(int)
    factory_df_cleaned['IntroKGM_total'] = factory_df_cleaned['Intro Pesée en Gr'] + factory_df_cleaned['Rectif ou Ajust Pesée en Gr']
    return factory_df_cleaned

def copy_formatting(src_cell, dest_cell):
    """
    Copy cell formatting from an Excel template.
    :param src_cell: The source cell from which to copy the formatting.
    :type src_cell: openpyxl.cell.cell.Cell
    :param dest_cell: The destination cell to which the formatting will be applied.
    :type dest_cell: openpyxl.cell.cell.Cell
    """
    # Copy the font, fill, border, and alignment
    if src_cell.has_style:
        dest_cell.font = copy(src_cell.font)
        dest_cell.border = copy(src_cell.border)
        dest_cell.fill = copy(src_cell.fill)
        dest_cell.number_format = src_cell.number_format

def highlight_above_threshold(value,threshold):
    """
    Highlight cells with an absolute value greater than a specified threshold.
    :param value: The value of the cell.
    :type value: float
    :param threshold: The chosen threshold value.
    :type threshold: float
    """
    is_above = np.abs(value) > threshold
    return ['background-color: red' if v else '' for v in is_above] 

def determine_MP_set(formula_df_sorted):
    """
    Determine the sorted set of raw materials for a given formula.
    :param formula_df_sorted: A DataFrame containing the sorted formula data.
    :type formula_df_sorted: pd.DataFrame
    """
    MP_set = formula_df_sorted['ComponentProductNo'].unique().tolist()
    MP_set = sorted(MP_set,key=str.lower)
    return MP_set

def generate_recommendation_summary(predictions_references_df , threshold_percentage):
    """
    Generate the recommendation summary for each formula .
    :param predictions_references_df: A DataFrame containing the new predicted references for the formulas.
    :type predictions_references_df: pd.DataFrame
    :param threshold_percentage: The threshold percentage for variation between the predicted relative percentage and the reference relative percentage of each raw material.
    :type threshold_percentage: integer
    """
    recommendation_summary_initial = ""
    entered = False
    for index , raw_material_row in predictions_references_df.iterrows():
        if np.abs(raw_material_row['difference_percentage_to_ref']) > threshold_percentage :
            entered = True
            recommendation_summary_initial+= "# "+raw_material_row["MP"]+" by "+str(raw_material_row["difference_percentage_to_ref"])+" % "
        
    if not entered:
        recommendation_summary_initial="-"
    return recommendation_summary_initial