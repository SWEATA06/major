def make_scaling_decision(
    predicted_cpu, 
    failure_prob, 
    uncertainty_score, 
    current_instances,
    latency=0.0,
    queue_length=0,
    cost_per_instance=0.05,
    budget_used=0.0,
    daily_budget=10.0,
    peak_hours=False
):
    """
    Advanced Cost-Aware and SLA-Driven Auto-scaling Decision Engine
    """
    # Overrides / SLA
    SLA_LATENCY_MAX = 500.0  # ms
    SLA_QUEUE_MAX = 1000
    FAILURE_RISK_EXTREME = 0.85
    FAILURE_RISK_HIGH = 0.70
    
    # Standard Thresholds
    CPU_HIGH = 75 if peak_hours else 85
    CPU_LOW = 30 if peak_hours else 40
    UNCERTAINTY_HIGH = 10.0
    
    target_instances = current_instances
    action = 'hold'
    reason = 'optimal'
    
    # 1. SLA Override Layer (Emergency)
    if latency > SLA_LATENCY_MAX or queue_length > SLA_QUEUE_MAX or failure_prob > FAILURE_RISK_EXTREME:
        action = 'urgent_scale_up'
        target_instances += 3
        reason = 'sla_violation_override'
        return action, target_instances, reason
        
    # 2. Risk Mitigation Layer
    if failure_prob > FAILURE_RISK_HIGH:
        action = 'scale_up'
        target_instances += 2
        reason = 'high_failure_risk'
        return action, target_instances, reason
        
    # 3. Cost-Aware Layer
    remaining_budget = daily_budget - budget_used
    budget_critical = remaining_budget < (daily_budget * 0.1)
    
    if predicted_cpu > CPU_HIGH:
        if budget_critical and not peak_hours:
            action = 'conservative_scaling'
            target_instances += 1
            reason = 'cpu_high_but_budget_critical'
        else:
            action = 'scale_up'
            target_instances += 2 if peak_hours else 1
            reason = 'cpu_high_normal'
            
    elif predicted_cpu < CPU_LOW and uncertainty_score <= UNCERTAINTY_HIGH:
        action = 'scale_down'
        if budget_critical:
            target_instances = max(1, current_instances - 2)
            reason = 'cpu_low_budget_saving'
        else:
            target_instances = max(1, current_instances - 1)
            reason = 'cpu_low_normal'
            
    elif uncertainty_score > UNCERTAINTY_HIGH:
        action = 'hold'
        reason = 'high_uncertainty_wait'
        
    return action, target_instances, reason
