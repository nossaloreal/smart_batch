
#!/usr/bin/env python
# coding: utf-8

import sys
#print(sys.path)
import pandas as pd
import numpy as np
import os


def processing(history_data_file, requested_formulas_data_file, reference_formulas_data_file, data_date, starting_date_grid_search, category_name_for_export_files = "pigment", raw_material_categories = ["PIGMENT","PATE PIG"]):
    current_path = os.getcwd()
    project_directory = os.path.dirname(os.path.abspath(current_path))
    history_data_file = history_data_file #"IA BPC Teinte v2.xlsm" #1st needed file
    requested_formulas_data_file = requested_formulas_data_file #"IA BPC Teinte Besoins.xlsm" #2nd needed file
    reference_formulas_data_file = reference_formulas_data_file #"composition des SOP avec type d'intro MP.xlsm" #3rd needed file

    sys.path.append(project_directory)
    from model.modules import Factory, Formula, Batch
    from model.outil import create_factory, upload_factory, determine_week_month_requested_formulas, check_datafiles, add_type_column_to_history_df, filter_history_df_selected_category, determine_selected_category_list, transform_formulas_prediction, transform_reference_df_selected_category, transform_reference_df, determine_selected_formulas_ids, determine_selected_category_percentages_df, determine_formulas_predictions_df_grouped, determine_reference_df_aux_selected_category_grouped

    data_date = data_date #"06-09-2024"
    SELECTED_MP_CATEGORIES = ["PIGMENT", "PATE PIG"]
    # SELECTED_MP_CATEGORIES = ["PHASE HUILEUSE"]
    category_name_for_export_files = "pigment"

    data_files_path = project_directory + "/data/data " + data_date + "/"
    temp_files_path = project_directory + "/temp files/temp files " + data_date + "/"
    plots_files_path = project_directory + "/plots files/plots files " + data_date + "/"
    results_files_path = project_directory + "/results files/results files " + data_date + "/"

    for path in [temp_files_path, plots_files_path, results_files_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder created at : {path}")
        else:
            print(f"Folder already exists at : {path}")

    df = pd.read_excel(history_data_file) #(data_files_path + history_data_filename)
    requested_formulas_df = pd.read_excel(requested_formulas_data_file)#(data_files_path + requested_formulas_data_filename)
    reference_df = pd.read_excel(reference_formulas_data_file) #(data_files_path + reference_formulas_data_filename)

    # check_datafiles(df,requested_formulas_df,reference_df)

    set_of_formulas_with_direct_introduction = set(df[df['UsageType'] == "Introduction directe"]['N° formule'].unique())

    df_aux_selected_category = filter_history_df_selected_category(df, SELECTED_MP_CATEGORIES)
    df_aux_selected_category = add_type_column_to_history_df(df_aux_selected_category, requested_formulas_df)
    selected_category_list = determine_selected_category_list(df, SELECTED_MP_CATEGORIES)
    selected_formulas_ids = determine_selected_formulas_ids(df_aux_selected_category, requested_formulas_df,
                                                            reference_df)

    print(len(selected_formulas_ids))

    factory_file_name = "factory_" + category_name_for_export_files + ".pkl"

    factory = create_factory(df_aux_selected_category, temp_files_path, factory_file_name)

    # factory = upload_factory(temp_files_path , factory_file_name)

    formulas_ids_with_high_cv = factory.select_formulas_of_cv_higher_than_threshold(0.175)
    formulas_ids_with_low_cv = factory.select_formulas_of_cv_lower_equal_than_threshold(0.2)
    adjusted_fomulas_ids_list = factory.get_adjusted_formulas_ids()
    no_unique_raw_material_composition_formulas_ids_list = factory.get_no_unique_raw_material_composition_formulas_ids()

    formulas_ids_to_plot = list(selected_formulas_ids)[:5]
    factory.plot_selected_formulas(formulas_ids_to_plot)

    plots_filename = "Formulas_plotted.pdf"
    # factory.plot_selected_formulas_to_pdf(formulas_ids_to_plot, plots_files_path + plots_filename)

    plots_filename = "Formulas_plotted_with_ref.pdf"
    # factory.plot_selected_formulas_with_ref_to_pdf(formulas_ids_to_plot, plots_files_path + plots_filename)

    THRESHOLD_ADJUSTEMENTS_ACCEPTED = 0
    NB_OF_RECENT_BATCHES_CONSIDERED = 3
    THRESHOLD_CV = 0.002
    MIN_ACCEPTED_BATCHES = 3

    stable_formulas = factory.search_of_stable_formulas(selected_formulas_ids=selected_formulas_ids,
                                                        threshold_adjustements_accepted=THRESHOLD_ADJUSTEMENTS_ACCEPTED,
                                                        nb_of_recent_batches_considered=NB_OF_RECENT_BATCHES_CONSIDERED,
                                                        threshold_cv=THRESHOLD_CV,
                                                        min_accepted_batches=MIN_ACCEPTED_BATCHES)

    # stable_formulas

    stable_formulas_ids_list = []
    for stable_formula in stable_formulas:
        stable_formulas_ids_list.append(list(stable_formula.keys())[0])

    selected_formulas_ids = selected_formulas_ids - set(stable_formulas_ids_list)

    last_batch_not_adjusted_formula = []
    for formula_id in factory.formulas_dict.keys():
        formula_object = factory.formulas_dict[formula_id]
        if not formula_object.batches_arr[-1].adjusted:
            last_batch_not_adjusted_formula.append(formula_id)

    stable_last_batch_formulas_ids_list = set(last_batch_not_adjusted_formula) & set(selected_formulas_ids)
    selected_formulas_ids = selected_formulas_ids - stable_last_batch_formulas_ids_list

    # LIMIT_INITIALIZATION_DATE = '2024-01-15'
    LIMIT_INITIALIZATION_DATE = '2024-03-06'

    param_ranges = {
        'nb_batches_to_remove': [1, 2],
        'min_nb_batches': [2, 3, 4, 5, 6, 7],
        'max_nb_adjustments': [5, 6, 7, 8, 9, 10, 11],
        'limit_initialization_date': [pd.Period(LIMIT_INITIALIZATION_DATE)],
        'weighted': [False, True],
        'initial_weight': [1, 1.2],
        'increase_rate': [0.2, 0.25, 0.3, 0.35]
    }

    for percentage_filtered_out_formulas_threshold in range(0, 101, 5):
        passed_boolean, best_params, best_metrics, formulas_predictions, metrics_MP, formulas_ids_filtered_out, selected_batches_dict = factory.grid_search(
            selected_formulas_ids, percentage_filtered_out_formulas_threshold, factory.function_metric, param_ranges)
        if passed_boolean:
            break

    print("Parameters used : ")
    print("nb of batches removed : ", best_params[0], "\t min nb of batches : ", best_params[1],
          "\t max nb of adjustments : ", best_params[2], "\t limit initialization date : ", best_params[3],
          "\t weighted : ", best_params[4], "\t initial_weight : ", best_params[5], "\t increase_rate : ",
          best_params[6], "")
    print("average metric : ", best_metrics[0], "\t percentage of formulas filtered out : ", best_metrics[1], "")

    print("List of formulas filtered out : ", formulas_ids_filtered_out, "")

    reference_df_included = reference_df[reference_df['productno'].isin(formulas_predictions.keys())]
    reference_df_aux = transform_reference_df(reference_df, formulas_predictions)
    reference_df_aux_selected_category = transform_reference_df_selected_category(reference_df_included,
                                                                                    selected_category_list)

    selected_category_percentages_df = determine_selected_category_percentages_df(reference_df_aux,
                                                                                selected_category_list)

    formulas_predictions_df = transform_formulas_prediction(formulas_predictions)
    formulas_predictions_df_grouped = determine_formulas_predictions_df_grouped(formulas_predictions_df,
                                                                                selected_category_percentages_df)
    reference_df_aux_selected_category_grouped = determine_reference_df_aux_selected_category_grouped(
        reference_df_aux_selected_category, selected_category_percentages_df)

    four_ten_days_requested_formulas_df_grouped, four_days_one_month_requested_formulas_df_grouped, earliest_date_after_four_days = determine_week_month_requested_formulas(
        requested_formulas_df, ref_date_str=data_date)

    parameters_metrics_variables = {'C3': '0',
                                    'C4': len(selected_formulas_ids),
                                    'C7': best_params[0],
                                    'C8': best_params[1],
                                    'C9': best_params[2],
                                    'C10': str(best_params[3]),
                                    'C11': best_params[4],
                                    'C12': best_params[5],
                                    'C13': best_params[6],
                                    'C16': str(round(best_metrics[1], 3)) + " %",
                                    'C17': '',
                                    'C20': best_metrics[0],
                                    'C23': len(stable_formulas),
                                    'C24': THRESHOLD_ADJUSTEMENTS_ACCEPTED,
                                    'C25': NB_OF_RECENT_BATCHES_CONSIDERED,
                                    'C26': MIN_ACCEPTED_BATCHES,
                                    'C27': THRESHOLD_CV
                                    }

    factory.export_predicted_and_ref_formulas_to_excel(
        results_files_path + "results_" + category_name_for_export_files + "_whole_history_colors.xlsx", 2,
        formulas_predictions_df_grouped, reference_df_aux_selected_category_grouped,
        four_ten_days_requested_formulas_df_grouped, four_days_one_month_requested_formulas_df_grouped,
        earliest_date_after_four_days, formulas_ids_filtered_out, stable_formulas_ids_list,
        stable_last_batch_formulas_ids_list, parameters_metrics_variables)

