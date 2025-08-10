#!/usr/bin/env python3
"""
Clean pie chart generation - so that they are not having overlapping texts for presentation sleides
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Create figures directory
figures_dir = Path('docs/figures')
figures_dir.mkdir(exist_ok=True)

def create_clean_data_distribution_pie():
    """Create a clean pie chart for mental health distribution."""
    
    # Data based on our MIMIC-IV results
    categories = ['Mood Disorders', 'Substance Mental', 'Psychotic Disorders', 
                 'Crisis Codes', 'Delirium Cognitive']
    counts = [2926, 6426, 493, 506, 239]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # Calculate percentages
    total_patients = sum(counts)
    percentages = [(count/total_patients)*100 for count in counts]
    
    # Create figure with extra space for legend
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Create clean pie chart - this approach works!
    wedges, texts = ax.pie(counts, colors=colors, startangle=90)
    
    ax.set_title('Proportion of Mental Health Conditions in MIMIC-IV Dataset\n(Total: 10,590 Patients)', 
                 fontsize=16, weight='bold', pad=20)
    
    # Create legend with colored rectangles
    legend_elements = []
    legend_labels = []
    
    for i, (category, percentage, count) in enumerate(zip(categories, percentages, counts)):
        legend_label = f"{category}: {percentage:.1f}% ({count:,} patients)"
        legend_labels.append(legend_label)
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], 
                                           edgecolor='black', linewidth=0.5))
    
    # Add legend below the pie chart
    ax.legend(legend_elements, legend_labels, 
             loc='center', bbox_to_anchor=(0.5, -0.05), 
             ncol=1, fontsize=12, frameon=True, 
             fancybox=True, shadow=True)
    
    # Save with proper spacing
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / 'data_distribution_pie.png', dpi=300, bbox_inches='tight')
    print("✅ Created: data_distribution_pie.png")
    plt.close()

def create_clean_model_parameters_pie():
    """Create a clean pie chart for model parameters distribution."""
    
    # Model Parameters Distribution
    layers = ['Text Encoder (Clinical BERT)', 'Fusion Layer', 'Vitals Encoder', 
             'Medication Encoder', 'SSL Head', 'Other Components']
    params = [25000000, 1000000, 1000000, 500000, 500000, 25090]  # Total: ~28M
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#DDA0DD']
    
    # Calculate percentages
    total_params = sum(params)
    percentages = [(param/total_params)*100 for param in params]
    
    # Create figure with extra space for legend
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Create clean pie chart - this approach works!
    wedges, texts = ax.pie(params, colors=colors, startangle=90)
    
    ax.set_title('Model Parameter Distribution\n(Total: 28M Parameters)', 
                 fontsize=16, weight='bold', pad=20)
    
    # Create legend with colored rectangles and detailed info
    legend_elements = []
    legend_labels = []
    
    for i, (layer, percentage, param_count) in enumerate(zip(layers, percentages, params)):
        # Format parameter count for readability
        if param_count >= 1000000:
            param_str = f"{param_count/1000000:.1f}M"
        elif param_count >= 1000:
            param_str = f"{param_count/1000:.0f}K"
        else:
            param_str = f"{param_count:,}"
        
        legend_label = f"{layer}: {percentage:.1f}% ({param_str} params)"
        legend_labels.append(legend_label)
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], 
                                           edgecolor='black', linewidth=0.5))
    
    # Add legend below the pie chart
    ax.legend(legend_elements, legend_labels, 
             loc='center', bbox_to_anchor=(0.5, -0.05), 
             ncol=1, fontsize=12, frameon=True, 
             fancybox=True, shadow=True)
    
    # Save with proper spacing
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_parameters_pie.png', dpi=300, bbox_inches='tight')
    print("✅ Created: model_parameters_pie.png")
    plt.close()

if __name__ == "__main__":
    print("Creating clean pie charts...")
    create_clean_data_distribution_pie()
    create_clean_model_parameters_pie()
)
