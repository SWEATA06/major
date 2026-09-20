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
    peak_hours=False,
    carbon_intensity=None,
    carbon_high_threshold=420.0
):
    """
    Advanced Cost-Aware, SLA-Driven and Carbon-Aware Auto-scaling Decision Engine.

    Decision layers, highest priority first:
      1. SLA override (emergency) - always wins, ignores cost and carbon.
      2. Failure risk mitigation - always wins over cost and carbon.
      3. Cost + carbon aware layer - only applies when there is no SLA/risk
         emergency, so we never trade an SLA breach for carbon savings.

    Carbon awareness (optional): when `carbon_intensity` (gCO2/kWh) is provided
    and exceeds `carbon_high_threshold`, the grid is "dirty". In that case we:
      - scale up more conservatively for merely-high (non-critical) CPU, and
      - trim harder when CPU is low,
    to shift/avoid compute during high-carbon periods. If carbon_intensity is
    None, behaviour is identical to the previous cost-aware engine.
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
    # CPU level considered a hard/critical spike that must be served regardless
    # of how dirty the grid is (still below the SLA override, but not deferrable).
    CPU_CRITICAL = 92

    carbon_dirty = carbon_intensity is not None and carbon_intensity > carbon_high_threshold
    
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
        
    # 3. Cost-Aware + Carbon-Aware Layer
    remaining_budget = daily_budget - budget_used
    budget_critical = remaining_budget < (daily_budget * 0.1)
    
    if predicted_cpu > CPU_HIGH:
        # Carbon deferral only for non-critical high CPU: if the grid is dirty
        # and load is high-but-safe, add fewer instances now (defer growth to a
        # cleaner window). Critical spikes are never deferred.
        if carbon_dirty and predicted_cpu < CPU_CRITICAL and not budget_critical:
            action = 'conservative_scaling'
            target_instances += 1
            reason = 'cpu_high_carbon_deferral'
        elif budget_critical and not peak_hours:
            action = 'conservative_scaling'
            target_instances += 1
            reason = 'cpu_high_but_budget_critical'
        else:
            action = 'scale_up'
            target_instances += 2 if peak_hours else 1
            reason = 'cpu_high_normal'
            
    elif predicted_cpu < CPU_LOW and uncertainty_score <= UNCERTAINTY_HIGH:
        action = 'scale_down'
        if budget_critical or carbon_dirty:
            # Trim harder when saving budget or when the grid is dirty.
            target_instances = max(1, current_instances - 2)
            reason = 'cpu_low_carbon_saving' if carbon_dirty and not budget_critical else 'cpu_low_budget_saving'
        else:
            target_instances = max(1, current_instances - 1)
            reason = 'cpu_low_normal'
            
    elif uncertainty_score > UNCERTAINTY_HIGH:
        action = 'hold'
        reason = 'high_uncertainty_wait'
        
    return action, target_instances, reason
