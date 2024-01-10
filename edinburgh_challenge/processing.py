def calculate_metric(results_dict):
    # Immediate cases
    immediate_completion_pct = results_dict["Completion Percentages"]["Immediate"]
    immediate_threshold_pct = results_dict["Threshold Compliance"]["Immediate"]
    immediate_deployment_time = results_dict["Mean Deployment Times"]["Immediate"]

    prompt_completion_pct = results_dict["Completion Percentages"]["Prompt"]
    prompt_threshold_pct = results_dict["Threshold Compliance"]["Prompt"]
    prompt_deployment_time = results_dict["Mean Deployment Times"]["Prompt"]

    standard_completion_pct = results_dict["Completion Percentages"]["Standard"]
    standard_threshold_pct = results_dict["Threshold Compliance"]["Standard"]
    standard_deployment_time = results_dict["Mean Deployment Times"]["Standard"]



    immediate_portion = (immediate_threshold_pct/100)
    prompt_portion = (prompt_threshold_pct/100)
    standard_portion = (standard_threshold_pct/100)

    # Immediate
    metric = (3*immediate_portion + 1.5*prompt_portion + 1*standard_portion)/(5.5)

    return metric
    
