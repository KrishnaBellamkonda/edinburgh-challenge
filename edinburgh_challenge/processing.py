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

def calculate_simulation_performance(results_dict):
    # Information from the results analysis
    immediate_completion_pct = results_dict["Completion Percentages"]["Immediate"]
    immediate_threshold_pct = results_dict["Threshold Compliance"]["Immediate"]

    prompt_completion_pct = results_dict["Completion Percentages"]["Prompt"]
    prompt_threshold_pct = results_dict["Threshold Compliance"]["Prompt"]

    standard_completion_pct = results_dict["Completion Percentages"]["Standard"]
    standard_threshold_pct = results_dict["Threshold Compliance"]["Standard"]

    mean_officer_hours  = results_dict["Mean Officer Hours"]

    # Rescaling these values
    immediate_completion_pct /= 100
    prompt_completion_pct /= 100
    standard_completion_pct /= 100

    immediate_threshold_pct /= 100
    prompt_threshold_pct /= 100
    standard_threshold_pct /= 100

    immediate_incompletion_pct = 1 - immediate_completion_pct
    prompt_incompletion_pct = 1- prompt_completion_pct
    standard_incompletion_pct = 1 - standard_completion_pct

    # Calculating the score

    # First factor - Incident resolved within threshold (Scale - 0 to 1)
    incident_within_threshold = (2*immediate_threshold_pct + 1.5*prompt_threshold_pct + 1*standard_threshold_pct)/(4.5)

    # Second factor - Officer utilisation
    # 8 hours per shift, 7 days in the simulation (Scale - 0 to 1)
    officer_utilisation = (mean_officer_hours)/(8*7 +1)

    # Third factor - Unresolved Incidents (Scale - 0 to 1)
    unresolved_incidents = ((6*immediate_incompletion_pct)+ 2*(prompt_incompletion_pct) + 1*(standard_incompletion_pct))/9

    # Total scale, (0 to 1)
    performance_metric = 0.8*incident_within_threshold + 0.2*officer_utilisation - unresolved_incidents*0.3
    return performance_metric
