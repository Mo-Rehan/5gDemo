from flask import Flask, render_template, jsonify, request
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict
import random

app = Flask(__name__)

# ============================================================================
# SOLUTION 1: Delay-Aware Scheduling (DAS)
# ============================================================================

@dataclass
class User:
    id: int
    service_type: str  # 'URLLC', 'eMBB', 'mMTC'
    instantaneous_rate: float
    avg_throughput: float
    packets: List[Dict]  # [{'remaining_time': int, 'size': float}]
    pdb: int  # Packet Delay Budget

def simulate_das(beta: float, num_slots: int = 100, num_users: int = 15, seed: int = None):
    """Simulate Delay-Aware Scheduling. Returns actual bytes transmitted per slot."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Initialize users: keep avg_throughput interpreted as bytes/sec-like (same unit as packet size)
    users = []
    for i in range(num_users):
        if i < 5:
            service_type = 'URLLC'; pdb = 5
        elif i < 10:
            service_type = 'eMBB'; pdb = 20
        else:
            service_type = 'mMTC'; pdb = 50

        users.append(User(
            id=i,
            service_type=service_type,
            instantaneous_rate=np.random.uniform(5, 20),  # normalized channel metric
            avg_throughput=10.0,  # historical throughput in same scale as packet size
            packets=[],
            pdb=pdb
        ))

    timeouts = {u.id: 0 for u in users}
    total_packets = {u.id: 0 for u in users}
    delays = {u.id: [] for u in users}
    throughput_history = []

    for slot in range(num_slots):
        # generate new packets
        for user in users:
            if np.random.random() < 0.3:
                size = np.random.uniform(1, 5)  # unit-consistent
                user.packets.append({'remaining_time': user.pdb, 'size': size})
                total_packets[user.id] += 1

        # age packets (decrement remaining_time)
        for user in users:
            for p in user.packets:
                p['remaining_time'] -= 1

        # compute priorities
        priorities = {}
        for user in users:
            # W1: per-user delay-criticality across its buffer
            w1 = 0.0
            total_buffer = sum(p['size'] for p in user.packets) if user.packets else 0.0
            if total_buffer > 0:
                for p in user.packets:
                    rt = p['remaining_time']
                    if rt > 0:
                        w1 += (p['size'] / total_buffer) * max(0.0, (1 - rt / user.pdb))
            # W2: PF-like metric: instantaneous channel metric / historical avg throughput
            w2 = user.instantaneous_rate / max(user.avg_throughput, 0.001)
            priorities[user.id] = beta * w1 + (1 - beta) * w2

        # select and serve one packet (one user)
        transmitted_bytes = 0.0
        if priorities:
            selected_user_id = max(priorities, key=priorities.get)
            selected_user = next(u for u in users if u.id == selected_user_id)
            if selected_user.packets:
                packet = selected_user.packets.pop(0)
                transmitted_bytes += packet['size']
                # record delay
                delay = selected_user.pdb - packet['remaining_time']
                delays[selected_user.id].append(delay)
                # update avg_throughput in same units as packet size
                selected_user.avg_throughput = 0.9 * selected_user.avg_throughput + 0.1 * packet['size']

        # check for timeouts
        for user in users:
            expired = [p for p in user.packets if p['remaining_time'] <= 0]
            timeouts[user.id] += len(expired)
            user.packets = [p for p in user.packets if p['remaining_time'] > 0]

        # record actual throughput (bytes transmitted this slot)
        throughput_history.append(transmitted_bytes)

        # update channel conditions
        for user in users:
            user.instantaneous_rate = max(1.0, user.instantaneous_rate + np.random.normal(0, 2))

    urllc_timeout_rate = sum(timeouts[u.id] for u in users if u.service_type == 'URLLC') / max(sum(total_packets[u.id] for u in users if u.service_type == 'URLLC'), 1) * 100
    avg_urllc_delay = np.mean([d for u in users if u.service_type == 'URLLC' for d in delays[u.id]]) if any(delays[u.id] for u in users if u.service_type == 'URLLC') else 0.0
    avg_throughput = np.mean(throughput_history) if throughput_history else 0.0

    return {
        'urllc_timeout_rate': urllc_timeout_rate,
        'avg_urllc_delay': avg_urllc_delay,
        'avg_throughput': avg_throughput,
        'throughput_history': throughput_history[-50:]
    }

# Add PF simulator for a true baseline
def simulate_pf(num_slots: int = 100, num_users: int = 15, seed: int = None):
    return simulate_das(beta=0.0, num_slots=num_slots, num_users=num_users, seed=seed)

# ============================================================================
# SOLUTION 2: Opportunistic Scheduling with CSI Smoothing
# ============================================================================

def simulate_csi_smoothing(alpha: float, num_slots: int = 100, num_users: int = 15):
    """Simulate Opportunistic Scheduling with EWMA CSI smoothing"""
    users = []
    for i in range(num_users):
        mobility = 'high' if i < 5 else 'low'
        users.append({
            'id': i,
            'mobility': mobility,
            'true_cqi': 10.0,
            'reported_cqi': 10.0,
            'smoothed_cqi': 10.0,
            'avg_throughput': 5.0,
            'retransmissions': 0,
            'successful_tx': 0
        })
    
    spectral_efficiency = []
    retransmission_rates = []
    
    for slot in range(num_slots):
        # Update true CQI (simulating channel changes)
        for user in users:
            change_rate = 2.0 if user['mobility'] == 'high' else 0.5
            user['true_cqi'] = max(1, user['true_cqi'] + np.random.normal(0, change_rate))
            
            # Add measurement noise
            noise = np.random.normal(0, 1.5) if user['mobility'] == 'high' else np.random.normal(0, 0.5)
            user['reported_cqi'] = max(1, user['true_cqi'] + noise)
            
            # EWMA smoothing
            user['smoothed_cqi'] = alpha * user['reported_cqi'] + (1 - alpha) * user['smoothed_cqi']
        
        # Calculate scheduling metric
        for user in users:
            user['metric'] = user['smoothed_cqi'] / max(user['avg_throughput'], 0.1)
        
        # Select best user
        selected_user = max(users, key=lambda u: u['metric'])
        
        # Transmission success based on CQI accuracy
        cqi_error = abs(selected_user['smoothed_cqi'] - selected_user['true_cqi'])
        success_prob = max(0.5, 1.0 - cqi_error / 10.0)
        
        if np.random.random() < success_prob:
            selected_user['successful_tx'] += 1
            selected_user['avg_throughput'] = 0.9 * selected_user['avg_throughput'] + 0.1 * selected_user['smoothed_cqi']
        else:
            selected_user['retransmissions'] += 1
        
        # Calculate metrics
        total_tx = sum(u['successful_tx'] + u['retransmissions'] for u in users)
        retrans_rate = sum(u['retransmissions'] for u in users) / max(total_tx, 1) * 100
        retransmission_rates.append(retrans_rate)
        
        avg_se = np.mean([u['smoothed_cqi'] for u in users])
        spectral_efficiency.append(avg_se)
    
    return {
        'spectral_efficiency': np.mean(spectral_efficiency),
        'retransmission_rate': np.mean(retransmission_rates),
        'se_history': spectral_efficiency[-50:],
        'retrans_history': retransmission_rates[-50:]
    }

# ============================================================================
# SOLUTION 3: Hierarchical Fragmentation-Aware Scheduling
# ============================================================================

def simulate_fragmentation_aware(clustering_enabled: bool, num_slots: int = 100):
    """Simulate Fragmentation-Aware Scheduling"""
    # Define spectrum fragments
    fragments = [
        {'id': 0, 'capacity': 50, 'center_freq': 3500},
        {'id': 1, 'capacity': 40, 'center_freq': 3600},
        {'id': 2, 'capacity': 30, 'center_freq': 3800},
        {'id': 3, 'capacity': 35, 'center_freq': 2600}
    ]
    
    num_users = 12
    utilization_history = []
    throughput_history = []
    urllc_compliance_history = []
    
    for slot in range(num_slots):
        # Generate user demands
        users = []
        for i in range(num_users):
            if i < 4:
                service_type = 'URLLC'
                demand = np.random.randint(5, 15)
                deadline = np.random.randint(3, 6)
                weight = 10.0
            elif i < 8:
                service_type = 'eMBB'
                demand = np.random.randint(15, 30)
                deadline = np.random.randint(10, 20)
                weight = 5.0
            else:
                service_type = 'mMTC'
                demand = np.random.randint(3, 10)
                deadline = np.random.randint(30, 50)
                weight = 1.0
            
            users.append({
                'id': i,
                'service_type': service_type,
                'demand': demand,
                'deadline': deadline,
                'weight': weight,
                'allocated': 0
            })
        
        # Calculate composite scores
        for user in users:
            fairness_factor = 1.0 / (user['allocated'] + 1)
            user['score'] = user['weight'] * (1.0 / user['deadline']) + fairness_factor
        
        users.sort(key=lambda u: u['score'], reverse=True)
        
        if clustering_enabled:
            # Cluster fragments by frequency proximity
            clusters = []
            for frag in fragments:
                added = False
                for cluster in clusters:
                    if any(abs(frag['center_freq'] - f['center_freq']) < 150 for f in cluster):
                        cluster.append(frag)
                        added = True
                        break
                if not added:
                    clusters.append([frag])
            
            # Allocate across clusters
            for user in users:
                remaining = user['demand'] - user['allocated']
                for cluster in clusters:
                    if remaining <= 0:
                        break
                    cluster_capacity = sum(f['capacity'] for f in cluster)
                    for frag in cluster:
                        if remaining <= 0:
                            break
                        available = frag['capacity']
                        allocated = min(remaining, available)
                        user['allocated'] += allocated
                        frag['capacity'] -= allocated
                        remaining -= allocated
        else:
            # Traditional per-fragment allocation
            for user in users:
                remaining = user['demand'] - user['allocated']
                for frag in fragments:
                    if remaining <= 0:
                        break
                    available = frag['capacity']
                    allocated = min(remaining, available)
                    user['allocated'] += allocated
                    frag['capacity'] -= allocated
                    remaining -= allocated
        
        # Calculate metrics
        original_total_capacity = 50 + 40 + 30 + 35  # or compute from a constant list
        total_allocated = sum(u['allocated'] for u in users)
        # utilization = allocated / total_capacity
        utilization = (total_allocated / original_total_capacity) * 100.0
        utilization_history.append(utilization)

        throughput = total_allocated
        throughput_history.append(throughput)

        urllc_users = [u for u in users if u['service_type'] == 'URLLC']
        urllc_compliance = sum(1 for u in urllc_users if u['allocated'] >= u['demand']) / max(len(urllc_users), 1) * 100
        urllc_compliance_history.append(urllc_compliance)
        
        # Reset fragment capacities
        fragments[0]['capacity'] = 50
        fragments[1]['capacity'] = 40
        fragments[2]['capacity'] = 30
        fragments[3]['capacity'] = 35
    
    return {
        'spectrum_utilization': np.mean(utilization_history),
        'avg_throughput': np.mean(throughput_history),
        'urllc_compliance': np.mean(urllc_compliance_history),
        'util_history': utilization_history[-50:],
        'throughput_history': throughput_history[-50:]
    }

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate/das', methods=['POST'])
def simulate_das_endpoint():
    data = request.json
    beta = float(data.get('beta', 0.7))
    results = simulate_das(beta)
    return jsonify(results)

@app.route('/simulate/csi', methods=['POST'])
def simulate_csi_endpoint():
    data = request.json
    alpha = float(data.get('alpha', 0.2))
    results = simulate_csi_smoothing(alpha)
    return jsonify(results)

@app.route('/simulate/fragmentation', methods=['POST'])
def simulate_fragmentation_endpoint():
    data = request.json
    clustering = bool(data.get('clustering', True))
    results = simulate_fragmentation_aware(clustering)
    return jsonify(results)

@app.route('/simulate/pf', methods=['POST'])
def simulate_pf_endpoint():
    data = request.json
    num_slots = int(data.get('num_slots', 100))
    results = simulate_pf(num_slots=num_slots)
    return jsonify(results)

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>5G RAN Resource Allocation - Interactive Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            color: #f0f0f0;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .solution-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .solution-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }
        .solution-title {
            font-size: 1.8em;
            color: #667eea;
            font-weight: bold;
        }
        .solution-description {
            color: #666;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        .controls {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .control-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #555;
        }
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        .value-display {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        button:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        button:active {
            transform: translateY(0);
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .comparison-note {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .improvement {
            color: #28a745;
            font-weight: bold;
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #667eea;
        }
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            font-weight: bold;
            margin: 20px 0;
        }
        .loading.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 5G RAN Resource Allocation Demo</h1>
        <p class="subtitle">Interactive Demonstration of QoS-Aware Scheduling Solutions</p>

        <!-- Solution 1: Delay-Aware Scheduling -->
        <div class="solution-card">
            <div class="solution-header">
                <div class="solution-title">Solution 1: Delay-Aware Scheduling (DAS)</div>
            </div>
            <div class="solution-description">
                This solution balances delay-critical URLLC traffic with fairness using a tunable parameter β. 
                When β=0, it behaves like traditional Proportional Fair scheduling. When β=1, it prioritizes delay-sensitive traffic.
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>
                        Trade-off Parameter (β): <span class="value-display" id="beta-value">0.7</span>
                    </label>
                    <input type="range" id="beta-slider" min="0" max="1" step="0.1" value="0.7">
                    <p style="font-size: 0.9em; color: #666; margin-top: 5px;">
                        β=0: Pure Proportional Fair (baseline) | β=1: Pure Delay-Aware
                    </p>
                </div>
                <button onclick="runDasSimulation()">Run Simulation</button>
            </div>

            <div class="loading" id="das-loading">Running simulation...</div>

            <div class="metrics" id="das-metrics" style="display: none;">
                <div class="metric-card">
                    <div class="metric-label">URLLC Timeout Rate</div>
                    <div class="metric-value" id="das-timeout">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg URLLC Delay (ms)</div>
                    <div class="metric-value" id="das-delay">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Throughput</div>
                    <div class="metric-value" id="das-throughput">-</div>
                </div>
            </div>

            <div class="comparison-note" id="das-comparison" style="display: none;">
                <strong>📊 Comparison vs Baseline (β=0):</strong>
                <div id="das-improvement"></div>
            </div>

            <div class="chart-container">
                <canvas id="das-chart"></canvas>
            </div>
        </div>

        <!-- Solution 2: CSI Smoothing -->
        <div class="solution-card">
            <div class="solution-header">
                <div class="solution-title">Solution 2: Opportunistic Scheduling with CSI Smoothing</div>
            </div>
            <div class="solution-description">
                Uses EWMA filtering to smooth noisy Channel State Information. Higher α values make the system more responsive 
                to recent measurements, while lower values provide more stability.
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>
                        Smoothing Factor (α): <span class="value-display" id="alpha-value">0.2</span>
                    </label>
                    <input type="range" id="alpha-slider" min="0.05" max="0.5" step="0.05" value="0.2">
                    <p style="font-size: 0.9em; color: #666; margin-top: 5px;">
                        α≈0: Maximum smoothing | α≈0.5: Minimal smoothing (like raw CSI)
                    </p>
                </div>
                <button onclick="runCsiSimulation()">Run Simulation</button>
            </div>

            <div class="loading" id="csi-loading">Running simulation...</div>

            <div class="metrics" id="csi-metrics" style="display: none;">
                <div class="metric-card">
                    <div class="metric-label">Spectral Efficiency</div>
                    <div class="metric-value" id="csi-efficiency">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Retransmission Rate (%)</div>
                    <div class="metric-value" id="csi-retrans">-</div>
                </div>
            </div>

            <div class="comparison-note" id="csi-comparison" style="display: none;">
                <strong>📊 Performance Analysis:</strong>
                <div id="csi-improvement"></div>
            </div>

            <div class="chart-container">
                <canvas id="csi-chart"></canvas>
            </div>
        </div>

        <!-- Solution 3: Fragmentation-Aware Scheduling -->
        <div class="solution-card">
            <div class="solution-header">
                <div class="solution-title">Solution 3: Hierarchical Fragmentation-Aware Scheduling</div>
            </div>
            <div class="solution-description">
                Manages non-contiguous spectrum fragments efficiently. With clustering enabled, fragments are grouped 
                for optimized allocation. Without clustering, fragments are treated independently (traditional approach).
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label style="display: flex; align-items: center; justify-content: space-between;">
                        <span>Fragment Clustering:</span>
                        <label class="toggle-switch">
                            <input type="checkbox" id="clustering-toggle" checked>
                            <span class="slider"></span>
                        </label>
                    </label>
                    <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                        OFF: Traditional per-fragment allocation | ON: Clustered optimization
                    </p>
                </div>
                <button onclick="runFragmentationSimulation()">Run Simulation</button>
            </div>

            <div class="loading" id="frag-loading">Running simulation...</div>

            <div class="metrics" id="frag-metrics" style="display: none;">
                <div class="metric-card">
                    <div class="metric-label">Spectrum Utilization (%)</div>
                    <div class="metric-value" id="frag-util">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Throughput</div>
                    <div class="metric-value" id="frag-throughput">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">URLLC Compliance (%)</div>
                    <div class="metric-value" id="frag-compliance">-</div>
                </div>
            </div>

            <div class="comparison-note" id="frag-comparison" style="display: none;">
                <strong>📊 Improvement with Clustering:</strong>
                <div id="frag-improvement"></div>
            </div>

            <div class="chart-container">
                <canvas id="frag-chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let dasChart, csiChart, fragChart;
        let baselineResults = null;

        // Update slider values
        document.getElementById('beta-slider').addEventListener('input', (e) => {
            document.getElementById('beta-value').textContent = e.target.value;
        });

        document.getElementById('alpha-slider').addEventListener('input', (e) => {
            document.getElementById('alpha-value').textContent = e.target.value;
        });

        // Solution 1: DAS
        async function runDasSimulation() {
            const beta = parseFloat(document.getElementById('beta-slider').value);
            
            document.getElementById('das-loading').classList.add('active');
            document.getElementById('das-metrics').style.display = 'none';
            
            try {
                const response = await fetch('/simulate/das', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({beta: beta})
                });
                
                const data = await response.json();
                
                // Get baseline if beta = 0 or first run
                if (beta === 0) {
                    baselineResults = data;
                }
                
                // Update metrics
                document.getElementById('das-timeout').textContent = data.urllc_timeout_rate.toFixed(2) + '%';
                document.getElementById('das-delay').textContent = data.avg_urllc_delay.toFixed(2);
                document.getElementById('das-throughput').textContent = data.avg_throughput.toFixed(2);
                
                // Show comparison
                if (beta > 0 && baselineResults) {
                    const safeDiv = (numer, denom) => denom === 0 ? 0 : (numer / denom * 100);
                    const timeoutImprovement = safeDiv(baselineResults.urllc_timeout_rate - data.urllc_timeout_rate, baselineResults.urllc_timeout_rate);
                    const delayImprovement = safeDiv(baselineResults.avg_urllc_delay - data.avg_urllc_delay, baselineResults.avg_urllc_delay);
                    
                    document.getElementById('das-improvement').innerHTML = `
                        <p>• URLLC Timeout Rate: <span class="improvement">${timeoutImprovement.toFixed(1)}% reduction</span></p>
                        <p>• URLLC Delay: <span class="improvement">${delayImprovement.toFixed(1)}% reduction</span></p>
                    `;
                    document.getElementById('das-comparison').style.display = 'block';
                }
                
                // Update chart
                if (dasChart) {
                    dasChart.destroy();
                }
                
                const ctx = document.getElementById('das-chart').getContext('2d');
                dasChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: data.throughput_history.length}, (_, i) => i),
                        datasets: [{
                            label: 'Throughput Over Time',
                            data: data.throughput_history,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top'
                            },
                            title: {
                                display: true,
                                text: `Throughput History (β=${beta})`
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Throughput'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time Slot'
                                }
                            }
                        }
                    }
                });
                
                document.getElementById('das-metrics').style.display = 'grid';
            } catch (error) {
                console.error('Error:', error);
                alert('Simulation failed. Please try again.');
            } finally {
                document.getElementById('das-loading').classList.remove('active');
            }
        }

        // Solution 2: CSI Smoothing
        async function runCsiSimulation() {
            const alpha = parseFloat(document.getElementById('alpha-slider').value);
            
            document.getElementById('csi-loading').classList.add('active');
            document.getElementById('csi-metrics').style.display = 'none';
            
            try {
                const response = await fetch('/simulate/csi', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({alpha: alpha})
                });
                
                const data = await response.json();
                
                // Update metrics
                document.getElementById('csi-efficiency').textContent = data.spectral_efficiency.toFixed(2);
                document.getElementById('csi-retrans').textContent = data.retransmission_rate.toFixed(2) + '%';
                
                // Show analysis
                let analysis = '';
                if (alpha < 0.15) {
                    analysis = '<p>• Heavy smoothing provides <span class="improvement">excellent stability</span> but may be slow to adapt to channel changes.</p>';
                } else if (alpha < 0.3) {
                    analysis = '<p>• <span class="improvement">Optimal balance</span> between stability and responsiveness. Recommended setting.</p>';
                } else {
                    analysis = '<p>• Low smoothing is more responsive but may suffer from noise-induced errors.</p>';
                }
                
                analysis += `<p>• Spectral Efficiency: ${data.spectral_efficiency.toFixed(2)} (higher is better)</p>`;
                analysis += `<p>• Retransmission Rate: ${data.retransmission_rate.toFixed(2)}% (lower is better)</p>`;
                
                document.getElementById('csi-improvement').innerHTML = analysis;
                document.getElementById('csi-comparison').style.display = 'block';
                
                // Update chart
                if (csiChart) {
                    csiChart.destroy();
                }
                
                const ctx = document.getElementById('csi-chart').getContext('2d');
                csiChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: data.se_history.length}, (_, i) => i),
                        datasets: [
                            {
                                label: 'Spectral Efficiency',
                                data: data.se_history,
                                borderColor: '#667eea',
                                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                fill: true,
                                tension: 0.4,
                                yAxisID: 'y'
                            },
                            {
                                label: 'Retransmission Rate (%)',
                                data: data.retrans_history,
                                borderColor: '#f093fb',
                                backgroundColor: 'rgba(240, 147, 251, 0.1)',
                                fill: true,
                                tension: 0.4,
                                yAxisID: 'y1'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top'
                            },
                            title: {
                                display: true,
                                text: `CSI Performance (α=${alpha})`
                            }
                        },
                        scales: {
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: {
                                    display: true,
                                    text: 'Spectral Efficiency'
                                }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                title: {
                                    display: true,
                                    text: 'Retransmission Rate (%)'
                                },
                                grid: {
                                    drawOnChartArea: false
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time Slot'
                                }
                            }
                        }
                    }
                });
                
                document.getElementById('csi-metrics').style.display = 'grid';
            } catch (error) {
                console.error('Error:', error);
                alert('Simulation failed. Please try again.');
            } finally {
                document.getElementById('csi-loading').classList.remove('active');
            }
        }

        // Solution 3: Fragmentation-Aware Scheduling
        let fragmentBaseline = null;
        
        async function runFragmentationSimulation() {
            const clustering = document.getElementById('clustering-toggle').checked;
            
            document.getElementById('frag-loading').classList.add('active');
            document.getElementById('frag-metrics').style.display = 'none';
            
            try {
                const response = await fetch('/simulate/fragmentation', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({clustering: clustering})
                });
                
                const data = await response.json();
                
                // Store baseline (without clustering)
                if (!clustering || !fragmentBaseline) {
                    if (!clustering) {
                        fragmentBaseline = data;
                    }
                }
                
                // Update metrics
                document.getElementById('frag-util').textContent = data.spectrum_utilization.toFixed(2) + '%';
                document.getElementById('frag-throughput').textContent = data.avg_throughput.toFixed(2);
                document.getElementById('frag-compliance').textContent = data.urllc_compliance.toFixed(2) + '%';
                
                // Show comparison
                if (clustering && fragmentBaseline) {
                    const utilImprovement = ((data.spectrum_utilization - fragmentBaseline.spectrum_utilization) / fragmentBaseline.spectrum_utilization * 100);
                    const throughputImprovement = ((data.avg_throughput - fragmentBaseline.avg_throughput) / fragmentBaseline.avg_throughput * 100);
                    
                    document.getElementById('frag-improvement').innerHTML = `
                        <p>• Spectrum Utilization: <span class="improvement">+${utilImprovement.toFixed(1)}% improvement</span></p>
                        <p>• Throughput: <span class="improvement">+${throughputImprovement.toFixed(1)}% improvement</span></p>
                        <p>• URLLC Compliance: <span class="improvement">${data.urllc_compliance.toFixed(1)}%</span></p>
                    `;
                    document.getElementById('frag-comparison').style.display = 'block';
                } else if (!clustering) {
                    document.getElementById('frag-improvement').innerHTML = `
                        <p>Baseline mode: Traditional per-fragment allocation. Enable clustering to see improvements!</p>
                    `;
                    document.getElementById('frag-comparison').style.display = 'block';
                }
                
                // Update chart
                if (fragChart) {
                    fragChart.destroy();
                }
                
                const ctx = document.getElementById('frag-chart').getContext('2d');
                fragChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: data.util_history.length}, (_, i) => i),
                        datasets: [
                            {
                                label: 'Spectrum Utilization (%)',
                                data: data.util_history,
                                borderColor: '#667eea',
                                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                fill: true,
                                tension: 0.4,
                                yAxisID: 'y'
                            },
                            {
                                label: 'Throughput',
                                data: data.throughput_history,
                                borderColor: '#f093fb',
                                backgroundColor: 'rgba(240, 147, 251, 0.1)',
                                fill: true,
                                tension: 0.4,
                                yAxisID: 'y1'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top'
                            },
                            title: {
                                display: true,
                                text: `Fragmentation-Aware Performance (Clustering: ${clustering ? 'ON' : 'OFF'})`
                            }
                        },
                        scales: {
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: {
                                    display: true,
                                    text: 'Utilization (%)'
                                }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                title: {
                                    display: true,
                                    text: 'Throughput'
                                },
                                grid: {
                                    drawOnChartArea: false
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Time Slot'
                                }
                            }
                        }
                    }
                });
                
                document.getElementById('frag-metrics').style.display = 'grid';
            } catch (error) {
                console.error('Error:', error);
                alert('Simulation failed. Please try again.');
            } finally {
                document.getElementById('frag-loading').classList.remove('active');
            }
        }

        // Run initial simulations on page load
        window.addEventListener('load', () => {
            runDasSimulation();
        });
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
import os
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    print("Starting 5G RAN Resource Allocation Demo...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)