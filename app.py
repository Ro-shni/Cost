from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from datetime import datetime
import plotly
import plotly.graph_objs as go
import plotly.express as px

app = Flask(__name__)

# Global variables to store data
data = {}

def load_data():
    """Load all CSV files into memory"""
    global data
    
    csv_files = {
        'node_comprehensive': 'node_analysis_corrected_fixed.csv',
        'over_utilized': 'over_utilized_nodes_fixed.csv',
        'imbalanced': 'imbalanced_nodes_fixed.csv',
        'waste_nodes': 'waste_nodes_fixed.csv',
        'pod_analysis': 'pod_resource_mismatch_analysis.csv',
        'cpu_heavy_pods': 'cpu_heavy_pods.csv',
        'memory_heavy_pods': 'memory_heavy_pods.csv',
        'nodepool_comprehensive': 'nodepool_comprehensive_analysis.csv',
        'namespace_comprehensive': 'namespace_comprehensive_analysis.csv',
        'bad_actors': 'bad_actors_comprehensive.csv'
    }
    
    for key, filename in csv_files.items():
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Normalize column names for consistency
                if key == 'bad_actors':
                    # Bad actors CSV has different column structure - normalize it
                    if 'nodepool_first' in df.columns:
                        df['nodepool'] = df['nodepool_first']
                    
                    # Map temporal columns to standard names for dashboard compatibility
                    column_mapping = {
                        'pod_cpu_req_cores_last': 'pod_cpu_req_cores',
                        'pod_mem_req_GB_last': 'pod_mem_req_GB',
                        'node_cpu_allocatable_first': 'node_cpu_allocatable',
                        'node_mem_allocatable_GB_first': 'node_mem_allocatable_GB',
                        'node_cpu_capacity_cores_first': 'node_cpu_capacity_cores',
                        'node_mem_capacity_GB_first': 'node_mem_capacity_GB',
                        'cpu_util_current': 'cpu_utilization_pct',
                        'mem_util_current': 'mem_utilization_pct',
                        'cpu_waste_current': 'cpu_unutilized',
                        'mem_waste_current': 'mem_unutilized_GB',
                        'cpu_over_current': 'cpu_over_requested',
                        'mem_over_current': 'mem_over_requested_GB'
                    }
                    
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            df[new_col] = df[old_col]
                    
                    # Add utilization pattern for bad actors
                    if 'utilization_pattern' not in df.columns:
                        def classify_utilization_pattern(row):
                            cpu_util = row.get('cpu_utilization_pct', 0)
                            mem_util = row.get('mem_utilization_pct', 0)
                            
                            if cpu_util > 100 and mem_util > 100:
                                return "Both Over-utilized"
                            elif cpu_util > 100 and mem_util < 80:
                                return "CPU Over-utilized, Memory Wasted"
                            elif mem_util > 100 and cpu_util < 80:
                                return "Memory Over-utilized, CPU Wasted"
                            elif cpu_util > 80 and mem_util > 80:
                                return "Both Highly Utilized"
                            elif cpu_util > 80 and mem_util < 80:
                                return "CPU Utilized, Memory Wasted"
                            elif mem_util > 80 and cpu_util < 80:
                                return "Memory Utilized, CPU Wasted"
                            elif cpu_util < 80 and mem_util < 80:
                                return "Both Under-utilized"
                            else:
                                return "Mixed Utilization"
                        
                        df['utilization_pattern'] = df.apply(classify_utilization_pattern, axis=1)
                
                elif key == 'nodepool_comprehensive':
                    # Map nodepool CSV columns to expected names
                    column_mapping = {
                        'cpu_waste_avg': 'cpu_unutilized',
                        'mem_waste_avg': 'mem_unutilized_GB',
                        'cpu_over_avg': 'cpu_over_requested',
                        'mem_over_avg': 'mem_over_requested_GB',
                        'cpu_util_avg': 'cpu_utilization_pct',
                        'mem_util_avg': 'mem_utilization_pct'
                    }
                    
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            df[new_col] = df[old_col]
                    
                    # Add missing columns expected by templates
                    if 'pod_cpu_waste' not in df.columns:
                        df['pod_cpu_waste'] = 0  # Pod-level waste not available at nodepool level
                    if 'pod_mem_waste_GB' not in df.columns:
                        df['pod_mem_waste_GB'] = 0
                    if 'pod_count' not in df.columns:
                        df['pod_count'] = 0  # Will be calculated if needed
                
                elif key == 'namespace_comprehensive':
                    # Map namespace CSV columns to expected names
                    column_mapping = {
                        'pod_cpu_req_cores_mean': 'pod_cpu_req_cores',
                        'pod_mem_req_GB_mean': 'pod_mem_req_GB',
                        'pod_cpu_waste_potential': 'pod_cpu_waste',
                        'pod_mem_waste_potential': 'pod_mem_waste_GB'
                    }
                    
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            df[new_col] = df[old_col]
                    
                    # Calculate average per pod metrics
                    if 'pod_count' in df.columns and df['pod_count'].sum() > 0:
                        df['avg_cpu_per_pod'] = df['pod_cpu_req_cores'] / df['pod_count']
                        df['avg_mem_per_pod'] = df['pod_mem_req_GB'] / df['pod_count']
                    else:
                        df['avg_cpu_per_pod'] = 0
                        df['avg_mem_per_pod'] = 0
                
                data[key] = df
                print(f"Loaded {filename}: {len(data[key])} rows")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                data[key] = pd.DataFrame()
        else:
            print(f"File not found: {filename}")
            data[key] = pd.DataFrame()

def get_summary_stats():
    """Calculate summary statistics for the dashboard"""
    stats = {}
    
    if 'node_comprehensive' in data and not data['node_comprehensive'].empty:
        node_df = data['node_comprehensive']
        stats.update({
            'total_nodes': len(node_df),
            'total_cpu_capacity': node_df['node_cpu_capacity_cores'].sum(),
            'total_mem_capacity': node_df['node_mem_capacity_GB'].sum(),
            'total_cpu_unutilized': node_df['cpu_unutilized'].sum(),
            'total_mem_unutilized': node_df['mem_unutilized_GB'].sum(),
            'total_cpu_over_requested': node_df['cpu_over_requested'].sum(),
            'total_mem_over_requested': node_df['mem_over_requested_GB'].sum(),
            'avg_cpu_utilization': node_df['cpu_utilization_pct'].mean(),
            'avg_mem_utilization': node_df['mem_utilization_pct'].mean()
        })
    
    if 'pod_analysis' in data and not data['pod_analysis'].empty:
        pod_df = data['pod_analysis']
        stats.update({
            'total_pods': len(pod_df),
            'total_pod_cpu_waste': pod_df['pod_cpu_waste'].sum(),
            'total_pod_mem_waste': pod_df['pod_mem_waste_GB'].sum(),
        })
    
    stats.update({
        'over_utilized_nodes': len(data.get('over_utilized', pd.DataFrame())),
        'imbalanced_nodes': len(data.get('imbalanced', pd.DataFrame())),
        'waste_nodes': len(data.get('waste_nodes', pd.DataFrame())),
        'bad_actors': len(data.get('bad_actors', pd.DataFrame())),
        'cpu_heavy_pods': len(data.get('cpu_heavy_pods', pd.DataFrame())),
        'memory_heavy_pods': len(data.get('memory_heavy_pods', pd.DataFrame()))
    })
    
    return stats

@app.route('/')
def dashboard():
    """Main dashboard page"""
    stats = get_summary_stats()
    return render_template('dashboard.html', stats=stats)

@app.route('/api/summary-charts')
def summary_charts():
    """API endpoint for summary charts"""
    charts = {}
    
    # Utilization pattern distribution
    if 'node_comprehensive' in data and not data['node_comprehensive'].empty:
        pattern_counts = data['node_comprehensive']['utilization_pattern'].value_counts()
        
        charts['utilization_patterns'] = {
            'data': [{
                'x': pattern_counts.index.tolist(),
                'y': pattern_counts.values.tolist(),
                'type': 'bar',
                'marker': {'color': '#3498db'}
            }],
            'layout': {
                'title': 'Node Utilization Patterns',
                'xaxis': {'title': 'Pattern'},
                'yaxis': {'title': 'Number of Nodes'}
            }
        }
    
    # Resource balance distribution
    if 'pod_analysis' in data and not data['pod_analysis'].empty:
        resource_counts = data['pod_analysis']['resource_pattern'].value_counts()
        
        charts['resource_patterns'] = {
            'data': [{
                'labels': resource_counts.index.tolist(),
                'values': resource_counts.values.tolist(),
                'type': 'pie',
                'marker': {'colors': ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c']}
            }],
            'layout': {
                'title': 'Pod Resource Patterns Distribution'
            }
        }
    
    return jsonify(charts)

@app.route('/nodes')
def nodes_view():
    """Nodes analysis page"""
    return render_template('nodes.html')

@app.route('/api/nodes')
def api_nodes():
    """API endpoint for nodes data"""
    node_type = request.args.get('type', 'all')
    search = request.args.get('search', '')
    nodepool = request.args.get('nodepool', '')
    
    if node_type == 'over_utilized':
        df = data.get('over_utilized', pd.DataFrame())
    elif node_type == 'imbalanced':
        df = data.get('imbalanced', pd.DataFrame())
    elif node_type == 'waste':
        df = data.get('waste_nodes', pd.DataFrame())
    elif node_type == 'bad_actors':
        df = data.get('bad_actors', pd.DataFrame())
    else:
        df = data.get('node_comprehensive', pd.DataFrame())
    
    if df.empty:
        return jsonify({'data': [], 'total': 0})
    
    # Apply filters
    if search:
        mask = df['node'].str.contains(search, case=False, na=False)
        df = df[mask]
    
    if nodepool:
        mask = df['nodepool'].str.contains(nodepool, case=False, na=False)
        df = df[mask]
    
    # Convert to records and handle NaN values
    records = df.fillna('').to_dict('records')
    
    return jsonify({
        'data': records,
        'total': len(records)
    })

@app.route('/pods')
def pods_view():
    """Pods analysis page"""
    return render_template('pods.html')

@app.route('/api/pods')
def api_pods():
    """API endpoint for pods data"""
    pod_type = request.args.get('type', 'all')
    search = request.args.get('search', '')
    namespace = request.args.get('namespace', '')
    nodepool = request.args.get('nodepool', '')
    
    if pod_type == 'cpu_heavy':
        df = data.get('cpu_heavy_pods', pd.DataFrame())
    elif pod_type == 'memory_heavy':
        df = data.get('memory_heavy_pods', pd.DataFrame())
    else:
        df = data.get('pod_analysis', pd.DataFrame())
    
    if df.empty:
        return jsonify({'data': [], 'total': 0})
    
    # Apply filters
    if search:
        mask = df['pod'].str.contains(search, case=False, na=False)
        df = df[mask]
    
    if namespace:
        mask = df['namespace'].str.contains(namespace, case=False, na=False)
        df = df[mask]
        
    if nodepool:
        mask = df['nodepool'].str.contains(nodepool, case=False, na=False)
        df = df[mask]
    
    # Convert to records and handle NaN values
    records = df.fillna('').to_dict('records')
    
    return jsonify({
        'data': records,
        'total': len(records)
    })

@app.route('/nodepools')
def nodepools_view():
    """Nodepools analysis page"""
    return render_template('nodepools.html')

@app.route('/api/nodepools')
def api_nodepools():
    """API endpoint for nodepools data"""
    df = data.get('nodepool_comprehensive', pd.DataFrame())
    
    if df.empty:
        return jsonify({'data': [], 'total': 0})
    
    search = request.args.get('search', '')
    if search:
        mask = df['nodepool'].str.contains(search, case=False, na=False)
        df = df[mask]
    
    # Convert to records and handle NaN values
    records = df.fillna('').to_dict('records')
    
    return jsonify({
        'data': records,
        'total': len(records)
    })

@app.route('/namespaces')
def namespaces_view():
    """Namespaces analysis page"""
    return render_template('namespaces.html')

@app.route('/api/namespaces')
def api_namespaces():
    """API endpoint for namespaces data"""
    df = data.get('namespace_comprehensive', pd.DataFrame())
    
    if df.empty:
        return jsonify({'data': [], 'total': 0})
    
    search = request.args.get('search', '')
    if search:
        mask = df['namespace'].str.contains(search, case=False, na=False)
        df = df[mask]
    
    # Convert to records and handle NaN values
    records = df.fillna('').to_dict('records')
    
    return jsonify({
        'data': records,
        'total': len(records)
    })

@app.route('/api/filters')
def api_filters():
    """API endpoint to get filter options"""
    filters = {}
    
    # Get nodepools from multiple sources
    nodepools = set()
    if 'node_comprehensive' in data and not data['node_comprehensive'].empty:
        nodepools.update(data['node_comprehensive']['nodepool'].dropna().unique().tolist())
    
    if 'bad_actors' in data and not data['bad_actors'].empty:
        if 'nodepool' in data['bad_actors'].columns:
            nodepools.update(data['bad_actors']['nodepool'].dropna().unique().tolist())
        elif 'nodepool_first' in data['bad_actors'].columns:
            nodepools.update(data['bad_actors']['nodepool_first'].dropna().unique().tolist())
    
    if nodepools:
        filters['nodepools'] = sorted(list(nodepools))
    
    # Get namespaces
    if 'pod_analysis' in data and not data['pod_analysis'].empty:
        filters['namespaces'] = sorted(data['pod_analysis']['namespace'].dropna().unique().tolist())
    
    return jsonify(filters)

@app.route('/api/temporal-metrics/<node_name>')
def api_temporal_metrics(node_name):
    """API endpoint to get temporal metrics for a specific node"""
    if 'bad_actors' not in data or data['bad_actors'].empty:
        return jsonify({'error': 'No temporal data available'})
    
    # Find the node in bad actors (which has full temporal data)
    node_data = data['bad_actors'][data['bad_actors']['node'] == node_name]
    
    if node_data.empty:
        # Try to find in main node data
        if 'node_comprehensive' in data and not data['node_comprehensive'].empty:
            node_data = data['node_comprehensive'][data['node_comprehensive']['node'] == node_name]
    
    if node_data.empty:
        return jsonify({'error': 'Node not found'})
    
    node_record = node_data.iloc[0].to_dict()
    
    # Extract temporal metrics if available
    temporal_metrics = {}
    temporal_fields = [
        'cpu_util_avg', 'cpu_util_std', 'cpu_util_min', 'cpu_util_max',
        'mem_util_avg', 'mem_util_std', 'mem_util_min', 'mem_util_max',
        'cpu_volatility', 'mem_volatility', 'cpu_trend', 'mem_trend',
        'cpu_consistency_score', 'mem_consistency_score',
        'observation_count', 'bad_actor_type', 'temporal_anomaly_score'
    ]
    
    for field in temporal_fields:
        if field in node_record:
            temporal_metrics[field] = node_record[field]
    
    return jsonify({
        'node': node_name,
        'temporal_metrics': temporal_metrics,
        'basic_info': {
            'cpu_utilization_pct': node_record.get('cpu_utilization_pct', 0),
            'mem_utilization_pct': node_record.get('mem_utilization_pct', 0),
            'nodepool': node_record.get('nodepool', ''),
            'utilization_pattern': node_record.get('utilization_pattern', ''),
            'bad_actor_type': node_record.get('bad_actor_type', 'Good Actor')
        }
    })

if __name__ == '__main__':
    print("Loading data...")
    load_data()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5005)
