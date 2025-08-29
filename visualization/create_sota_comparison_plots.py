#!/usr/bin/env python3
"""
Create SOTA Comparison Plots
============================

This script creates comprehensive visualization comparing our GTF-shPLRNN 
with state-of-the-art methods based on experimental results.

Author: Master Thesis Project
Date: 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load comparison results"""
    with open('../../results/sota_comparison_results_manual.json', 'r') as f:
        results = json.load(f)
    return results

def create_comprehensive_sota_plots():
    """Create comprehensive SOTA comparison plots"""
    results = load_results()
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    models = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Professional color scheme
    
    # 1. Test Performance Comparison (Micro F1)
    ax1 = axes[0, 0]
    micro_f1s = [results[m]['test_metrics']['micro_f1'] for m in models]
    
    bars = ax1.bar(models, micro_f1s, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, micro_f1s)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        # Highlight our method
        if models[i] == 'GTF_shPLRNN':
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)
    
    ax1.set_ylabel('Test Micro F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('📊 Test Performance Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(micro_f1s) * 1.2)
    
    # 2. Hamming Loss Comparison
    ax2 = axes[0, 1]
    hamming_losses = [results[m]['test_metrics']['hamming_loss'] for m in models]
    
    bars = ax2.bar(models, hamming_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for i, (bar, value) in enumerate(zip(bars, hamming_losses)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        if models[i] == 'GTF_shPLRNN':
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)
    
    ax2.set_ylabel('Hamming Loss', fontsize=12, fontweight='bold')
    ax2.set_title('📉 Hamming Loss Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Parameter Efficiency
    ax3 = axes[0, 2]
    param_counts = [results[m]['total_params'] for m in models]
    test_f1s = [results[m]['test_metrics']['micro_f1'] for m in models]
    
    for i, model in enumerate(models):
        marker = 'D' if model == 'GTF_shPLRNN' else 'o'
        size = 200 if model == 'GTF_shPLRNN' else 150
        ax3.scatter(param_counts[i], test_f1s[i], 
                   s=size, c=colors[i], alpha=0.8, marker=marker, 
                   edgecolors='black', linewidth=2)
        
        # Add labels
        ax3.annotate(model, (param_counts[i], test_f1s[i]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
    
    ax3.set_xlabel('Parameter Count', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Test Micro F1', fontsize=12, fontweight='bold')
    ax3.set_title('🎯 Parameter Efficiency\n(Top-Left is Best)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Detailed Metrics Radar Chart
    ax4 = axes[1, 0]
    
    # Select GTF-shPLRNN and best competitor for detailed comparison
    best_competitor = max([m for m in models if m != 'GTF_shPLRNN'], 
                         key=lambda x: results[x]['test_metrics']['micro_f1'])
    
    metrics = ['Micro F1', 'Macro F1', 'Precision', 'Recall', '1-Hamming']
    
    gtf_values = [
        results['GTF_shPLRNN']['test_metrics']['micro_f1'],
        results['GTF_shPLRNN']['test_metrics']['macro_f1'],
        results['GTF_shPLRNN']['test_metrics']['micro_precision'],
        results['GTF_shPLRNN']['test_metrics']['micro_recall'],
        1 - results['GTF_shPLRNN']['test_metrics']['hamming_loss']
    ]
    
    competitor_values = [
        results[best_competitor]['test_metrics']['micro_f1'],
        results[best_competitor]['test_metrics']['macro_f1'],
        results[best_competitor]['test_metrics']['micro_precision'],
        results[best_competitor]['test_metrics']['micro_recall'],
        1 - results[best_competitor]['test_metrics']['hamming_loss']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, gtf_values, width, label='GTF-shPLRNN (Ours)', 
                    color='#C73E1D', alpha=0.8, edgecolor='black')
    bars2 = ax4.bar(x + width/2, competitor_values, width, label=f'{best_competitor}', 
                    color='#2E86AB', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title(f'🔍 Detailed Comparison:\nGTF-shPLRNN vs {best_competitor}', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=20)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Training Efficiency
    ax5 = axes[1, 1]
    epochs_trained = [results[m]['epochs_trained'] for m in models]
    best_val_f1s = [results[m]['best_val_f1'] for m in models]
    
    for i, model in enumerate(models):
        marker = 'D' if model == 'GTF_shPLRNN' else 'o'
        size = 200 if model == 'GTF_shPLRNN' else 150
        ax5.scatter(epochs_trained[i], best_val_f1s[i], 
                   s=size, c=colors[i], alpha=0.8, marker=marker,
                   edgecolors='black', linewidth=2)
        
        ax5.annotate(model, (epochs_trained[i], best_val_f1s[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.3))
    
    ax5.set_xlabel('Epochs Trained', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Best Validation F1', fontsize=12, fontweight='bold')
    ax5.set_title('⚡ Training Efficiency\n(Top-Left is Best)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Overall Ranking Summary
    ax6 = axes[1, 2]
    
    # Calculate composite scores
    composite_scores = {}
    for model in models:
        metrics = results[model]['test_metrics']
        # Weighted composite: F1 (0.5) + (1-Hamming) (0.3) + Precision (0.2)
        composite = (0.5 * metrics['micro_f1'] + 
                    0.3 * (1 - metrics['hamming_loss']) + 
                    0.2 * metrics['micro_precision'])
        composite_scores[model] = composite
    
    # Sort by composite score
    sorted_models = sorted(models, key=lambda x: composite_scores[x], reverse=True)
    sorted_scores = [composite_scores[m] for m in sorted_models]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(sorted_models))
    bars = ax6.barh(y_pos, sorted_scores, 
                    color=[colors[models.index(m)] for m in sorted_models], 
                    alpha=0.8, edgecolor='black', linewidth=1)
    
    # Highlight our method
    for i, model in enumerate(sorted_models):
        if model == 'GTF_shPLRNN':
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_scores)):
        width = bar.get_width()
        ax6.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(sorted_models, fontweight='bold')
    ax6.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax6.set_title('🏆 Overall Ranking\n(Composite Score)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('../../results/sota_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 SOTA comparison plots saved as 'sota_comparison_comprehensive.png'")

def create_summary_table():
    """Create a summary table of results"""
    results = load_results()
    
    print("\n" + "="*80)
    print("🎯 SOTA METHODS COMPARISON SUMMARY")
    print("="*80)
    
    print(f"{'Model':<15} {'Test F1':<10} {'Hamming':<10} {'Params':<12} {'Epochs':<8} {'Ranking'}")
    print("-" * 80)
    
    # Sort by test F1
    sorted_models = sorted(results.keys(), 
                          key=lambda x: results[x]['test_metrics']['micro_f1'], 
                          reverse=True)
    
    for i, model in enumerate(sorted_models, 1):
        data = results[model]
        f1 = data['test_metrics']['micro_f1']
        hamming = data['test_metrics']['hamming_loss']
        params = data['total_params']
        epochs = data['epochs_trained']
        
        marker = " 🏆" if model == 'GTF_shPLRNN' else ""
        print(f"{model:<15} {f1:<10.4f} {hamming:<10.4f} {params:<12,} {epochs:<8} #{i}{marker}")
    
    print("-" * 80)
    
    # Key findings
    print("\n🔍 KEY FINDINGS:")
    best_model = max(results.keys(), key=lambda x: results[x]['test_metrics']['micro_f1'])
    gtf_f1 = results['GTF_shPLRNN']['test_metrics']['micro_f1']
    best_f1 = results[best_model]['test_metrics']['micro_f1']
    
    if best_model == 'GTF_shPLRNN':
        print("✅ GTF-shPLRNN achieves the BEST performance among all methods!")
        resnet_f1 = results['ResNet_1D']['test_metrics']['micro_f1']
        improvement = ((gtf_f1 - resnet_f1) / resnet_f1) * 100
        print(f"✅ Our method outperforms ResNet-1D by {improvement:.1f}%")
    else:
        gap = best_f1 - gtf_f1
        print(f"📊 GTF-shPLRNN ranks #{sorted_models.index('GTF_shPLRNN') + 1}")
        print(f"📊 Performance gap to best: {gap:.4f}")
    
    # Parameter efficiency
    gtf_params = results['GTF_shPLRNN']['total_params']
    resnet_params = results['ResNet_1D']['total_params']
    efficiency = resnet_params / gtf_params
    print(f"⚡ GTF-shPLRNN uses {efficiency:.1f}x fewer parameters than ResNet-1D")
    
    print("="*80)

if __name__ == "__main__":
    create_comprehensive_sota_plots()
    create_summary_table()