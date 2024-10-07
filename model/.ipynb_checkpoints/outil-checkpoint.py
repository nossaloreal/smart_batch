from modules import Factory 
import pickle
from datetime import datetime , timedelta
import os
import pandas as pd

def check_datafiles(df , requested_formulas_df , reference_df):
    """
    Print detailed information about the provided data files.
    :param df: A DataFrame containing the historical batches of formulas.
    :type df: pd.DataFrame
    :param requested_formulas_df: A DataFrame containing the requested formulas to be produced.
    :type requested_formulas_df: pd.DataFrame
    :param reference_df: A DataFrame containing the reference formulas.
    :type reference_df: pd.DataFrame
    """
    df_aux = df[df['NSQ'] == 'NSQ3']
    df_formulas_set = set(df_aux['N° formule'])
    requested_df_formulas_set = set(requested_formulas_df['OP_ARTICLE'])
    reference_df_formulas_set = set(reference_df['productno'])
    selected_formulas_ids =  (requested_df_formulas_set & reference_df_formulas_set) & df_formulas_set
    requested_formulas_with_no_reference = requested_df_formulas_set - (requested_df_formulas_set & reference_df_formulas_set)
    requested_formulas_with_no_history = requested_df_formulas_set - (requested_df_formulas_set & df_formulas_set)
    print("Total number of requested formulas :" , len(requested_df_formulas_set) , "\n")
    print("Number of requested formulas with both history and reference data :" , len(selected_formulas_ids) )
    print("Requested formula ids with both history and reference data :" , selected_formulas_ids , "\n")
    print("Number of requested formulas with no reference data :" , len(requested_formulas_with_no_reference) )
    print("Requested formula ids with no reference data :" , requested_formulas_with_no_reference , "\n")
    print("Number of requested formulas with no history data :" , len(requested_formulas_with_no_history) )
    print("Requested formula ids with no history data :" , requested_formulas_with_no_history , "\n")

def add_type_column_to_history_df(df_aux , requested_formulas_df):
    """
    Add a formula type column to the detailed formulas history DataFrame.
    :param df_aux: A DataFrame containing the historical batches of formulas.
    :type df_aux: pd.DataFrame
    :param requested_formulas_df: A DataFrame containing the formulas requested for production.
    :type requested_formulas_df: pd.DataFrame
    """
    requested_formulas_df_aux = requested_formulas_df[["Type","OP_ARTICLE"]]
    requested_formulas_df_aux = requested_formulas_df_aux.rename(columns={"OP_ARTICLE":"N° formule"})
    t_df = requested_formulas_df_aux.drop_duplicates()

    # Step 1: Group by 'id' and filter out those with different 'value'
    ids_with_different_values = t_df.groupby('N° formule')['Type'].nunique()
    ids_to_drop = ids_with_different_values[ids_with_different_values > 1].index
    # Step 2: Drop rows with those ids
    t_df = t_df[~t_df['N° formule'].isin(ids_to_drop)]
    # Step 3: Add rows with id and value="-"
    replacement_rows = pd.DataFrame({'N° formule': ids_to_drop, 'Type': '-'})
    t_df = pd.concat([t_df, replacement_rows], ignore_index=True).sort_values(by='N° formule')

    df_aux = pd.merge(df_aux,t_df , on="N° formule", how="left")
    df_aux["Type"] = df_aux["Type"].fillna("na")
    return df_aux

def filter_history_df(df):
    """
    Filtering the formulas history DataFrame.
    :param df: A DataFrame containing the historical batches of formulas.
    :type df: pd.DataFrame
    """
    set_of_formulas_with_direct_introduction = set(df[df['UsageType']=="Introduction directe"]['N° formule'].unique())
    df_aux = df[(~(df['N° formule'].isin(list(set_of_formulas_with_direct_introduction))))&(df['NSQ'] == 'NSQ3')]
    return df_aux

def filter_history_df_selected_category(df , selected_MP_categories):
    """
    Filter the formulas history DataFrame based on selected raw material categories.
    :param df: A DataFrame containing the historical batches of formulas.
    :type df: pd.DataFrame
    :param selected_MP_categories: A list of raw material categories to be considered for filtering.
    :type selected_MP_categories: list
    """
    set_of_formulas_with_direct_introduction = set(df[df['UsageType']=="Introduction directe"]['N° formule'].unique())
    df_aux_selected_category = df[(~(df['N° formule'].isin(list(set_of_formulas_with_direct_introduction))))&(df['NSQ'] == 'NSQ3')&(df['Type MP'].isin(selected_MP_categories))]
    return df_aux_selected_category

def determine_selected_category_list(df , selected_MP_categories):
    """
    Determine the list of unique raw materials that belong to the selected categories.
    :param df: A DataFrame containing the historical batches of formulas.
    :type df: pd.DataFrame
    :param selected_MP_categories: A list of raw material categories to filter by.
    :type selected_MP_categories: list
    """
    set_of_formulas_with_direct_introduction = set(df[df['UsageType']=="Introduction directe"]['N° formule'].unique())
    selected_category_list = df[(~(df['N° formule'].isin(list(set_of_formulas_with_direct_introduction))))&(df['NSQ'] == 'NSQ3')&(df['Type MP'].isin(selected_MP_categories))]["ComponentProductNo"].unique()
    return selected_category_list

def determine_selected_formulas_ids(df_aux , requested_formulas_df, reference_df):
    """
    Determine the list of selected fromulas.
    :param df_aux: A DataFrame containing the historical batches of formulas.
    :type df_aux: pd.DataFrame
    :param requested_formulas_df: A DataFrame containing the requested formulas to be produced.
    :type requested_formulas_df: pd.DataFrame
    :param reference_df: A DataFrame containing the reference formulas.
    :type reference_df: pd.DataFrame
    """
    df_formulas_set = set(df_aux['N° formule'])
    requested_df_formulas_set = set(requested_formulas_df['OP_ARTICLE'])
    reference_df_formulas_set = set(reference_df['productno'])
    selected_formulas_ids =  (requested_df_formulas_set & reference_df_formulas_set) & df_formulas_set
    return selected_formulas_ids

def create_factory(df_factory , temp_files_path ,factory_file_name):
    """
    Create Factory object.
    :param df_factory: A DataFrame containing the historical batches of formulas.
    :type df_factory: pd.DataFrame
    :param temp_files_path: directory path where to save the Factory Object in a pickle file.
    :type temp_files_path: string
    :param factory_file_name: Factory pickle filename.
    :type factory_file_name: string
    """
    factory = Factory(df_factory)
    file_name_exists = False
    if os.path.exists(temp_files_path+factory_file_name):
        os.remove(temp_files_path+factory_file_name)
        file_name_exists = True
    with open(temp_files_path+factory_file_name, "wb") as f:
        pickle.dump(factory, f)
    if file_name_exists :
        print(f"File replaced: {factory_file_name}")
    else:
        print(f"File created: {factory_file_name}")
    return factory

def upload_factory(temp_files_path , factory_file_name):
    """
    Load a Factory object from a pickle file.
    :param temp_files_path: Directory path where the pickle file containing the Factory object is located.
    :type temp_files_path: str
    :param factory_file_name: The name of the pickle file containing the Factory object.
    :type factory_file_name: str
    """
    with open(temp_files_path+factory_file_name, "rb") as f:
        factory = pickle.load(f)
    return factory

def transform_formulas_prediction(formulas_predictions):
    """
    Transform the formulas predictions into a dataframe.
    :param formulas_predictions: A DataFrame containing the new predicted references for the formulas.
    :type formulas_predictions: pd.DataFrame
    """
    formulas_predictions_flattened= [(outer_key, inner_key, value)
                 for outer_key, inner_dict in formulas_predictions.items()
                 for inner_key, value in inner_dict.items()]
    # Create the DataFrame
    formulas_predictions_df = pd.DataFrame(formulas_predictions_flattened, columns=['productno', 'MP', 'formulas_predictions'])
    return formulas_predictions_df

def transform_reference_df(reference_df , formulas_predictions):
    """
    Transform the reference formulas DataFrame to ensure compatibility.
    :param reference_df: A DataFrame containing the reference formulas.
    :type reference_df: pd.DataFrame
    :param formulas_predictions: A DataFrame containing the predicted references for the formulas.
    :type formulas_predictions: pd.DataFrame
    """
    reference_df_aux = reference_df[reference_df['productno'].isin(formulas_predictions.keys())]
    reference_df_aux = reference_df_aux[reference_df_aux['Integration_Method'] != 'Introduction directe']

    reference_df_aux = reference_df_aux.groupby(['productno' ,'Produit']).agg({'Quantity': 'sum'}).reset_index()
    reference_df_aux['total_quantity'] = reference_df_aux.groupby('productno' )['Quantity'].transform('sum')
    reference_df_aux['formulas_ref'] = (reference_df_aux['Quantity'] / reference_df_aux['total_quantity'])*100
    reference_df_aux['formulas_ref'] = reference_df_aux['formulas_ref'].round(4)
    return reference_df_aux

def transform_reference_df_selected_category(reference_df_included , selected_category_list):
    """
    Transform the reference formulas DataFrame to ensure compatibility.
    :param reference_df: A DataFrame containing the reference formulas.
    :type reference_df: pd.DataFrame
    :param selected_category_list: A list of raw material categories to be considered for filtering.
    :type selected_category_list: list
    """
    reference_df_aux_selected_category = reference_df_included[(reference_df_included['Integration_Method'] != 'Introduction directe')&(reference_df_included['Produit'].isin(selected_category_list))]
    reference_df_aux_selected_category = reference_df_aux_selected_category.groupby(['productno' ,'Produit']).agg({'Quantity': 'sum'}).reset_index()
    reference_df_aux_selected_category['total_quantity'] = reference_df_aux_selected_category.groupby('productno' )['Quantity'].transform('sum')
    reference_df_aux_selected_category['formulas_ref'] = (reference_df_aux_selected_category['Quantity'] / reference_df_aux_selected_category['total_quantity'])*100
    reference_df_aux_selected_category['formulas_ref'] = reference_df_aux_selected_category['formulas_ref'].round(4)
    return reference_df_aux_selected_category
    
def determine_selected_category_percentages_df(reference_df_aux , selected_category_list):
    """
    Determine the composition percentages DataFrame for raw material.
    :param reference_df_aux: A DataFrame containing the reference formulas.
    :type reference_df_aux: pd.DataFrame
    :param selected_category_list: A list of raw material categories to be considered for filtering.
    :type selected_category_list: list
    """
    selected_category_percentages_df = reference_df_aux[~(reference_df_aux['Produit'].isin(selected_category_list))].groupby('productno')['formulas_ref'].sum().reset_index()
    selected_category_percentages_df.rename(columns={'formulas_ref': 'selected_category_percentage'},inplace=True)
    selected_category_percentages_df["selected_category_percentage"] = 100 - selected_category_percentages_df["selected_category_percentage"] 
    return selected_category_percentages_df

def determine_formulas_predictions_df_grouped(formulas_predictions_df , selected_category_percentages_df):
    formulas_predictions_df_grouped = pd.merge(formulas_predictions_df,selected_category_percentages_df,on="productno")
    formulas_predictions_df_grouped["formulas_predictions"] = formulas_predictions_df_grouped["formulas_predictions"]*(formulas_predictions_df_grouped["selected_category_percentage"]/100)
    return formulas_predictions_df_grouped

def determine_reference_df_aux_selected_category_grouped(reference_df_aux_selected_category , selected_category_percentages_df):
    reference_df_aux_selected_category_grouped = pd.merge(reference_df_aux_selected_category,selected_category_percentages_df,on="productno")
    reference_df_aux_selected_category_grouped["formulas_ref"] = reference_df_aux_selected_category_grouped["formulas_ref"]*(reference_df_aux_selected_category_grouped["selected_category_percentage"]/100)
    return reference_df_aux_selected_category_grouped

def determine_week_month_requested_formulas(requested_formulas_df , ref_date_str):
    ref_datetime = datetime.strptime(ref_date_str , "%d-%m-%Y")
    ref_datetime_plus_four_days = ref_datetime + timedelta(days=4)
    ref_datetime_plus_ten_days = ref_datetime + timedelta(days=10)
    ref_datetime_plus_four_days_str = ref_datetime_plus_four_days.strftime("%Y-%m-%d")
    ref_datetime_plus_ten_days_str = ref_datetime_plus_ten_days.strftime("%Y-%m-%d")

    year = ref_datetime.year + (ref_datetime.month + 1) // 12
    month = (ref_datetime.month + 1) % 12
    if month == 0:
        month = 12
    day = min(ref_datetime.day, [31,
                        29 if year % 4 == 0 and not (year % 100 == 0 and year % 400 != 0) else 28,
                        31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    ref_datetime_plus_month = datetime(year, month, day)
    ref_datetime_plus_month_str = ref_datetime_plus_month.strftime("%Y-%m-%d")

    four_ten_days_requested_formulas_df=requested_formulas_df[(requested_formulas_df['OP_DATE_DEBUT']>=ref_datetime_plus_four_days_str) & (requested_formulas_df['OP_DATE_DEBUT']<=ref_datetime_plus_ten_days_str)]
    four_days_one_month_requested_formulas_df=requested_formulas_df[(requested_formulas_df['OP_DATE_DEBUT']>=ref_datetime_plus_four_days_str)&(requested_formulas_df['OP_DATE_DEBUT']<=ref_datetime_plus_month_str)]
    four_days_requested_formuals_df = requested_formulas_df[(requested_formulas_df['OP_DATE_DEBUT']>=ref_datetime_plus_four_days_str) ]

    earliest_date_after_four_days = four_days_requested_formuals_df.groupby(['OP_ARTICLE'])['OP_DATE_DEBUT'].min().reset_index()
    four_ten_days_requested_formulas_df_grouped = four_ten_days_requested_formulas_df.groupby(['OP_ARTICLE'])['OP_ORDRE'].count().reset_index()
    four_days_one_month_requested_formulas_df_grouped = four_days_one_month_requested_formulas_df.groupby(['OP_ARTICLE'])['OP_ORDRE'].count().reset_index()
    
    return four_ten_days_requested_formulas_df_grouped , four_days_one_month_requested_formulas_df_grouped , earliest_date_after_four_days