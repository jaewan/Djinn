import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- PLOT CONFIGURATION ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'hatch.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})

# --- PROFESSIONAL ACADEMIC COLORS ---
C_COMPUTE  = '#004488'   # Dark Blue
C_STREAM   = '#66CCEE'   # Light Cyan/Blue
C_TRANSFER = '#EE6677'   # Burnt Orange/Red
C_OVERHEAD = '#BBBBBB'   # Neutral Grey
C_TEXT     = '#333333'

def plot_osdi_figure():
    # --- DATA ---
    ds_compute = 0.7
    ds_transfer = 35.5
    ds_total = 36.2
    ds_data_gb = 24.2

    dj_compute = 0.7
    dj_stream_duration = 0.4
    dj_overhead = 0.45
    dj_total = 1.15
    dj_data_gb = 6.0

    # --- SETUP BROKEN AXIS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 5.5), 
                                   gridspec_kw={'height_ratios': [1, 2.5]})
    
    bar_width = 0.5
    
    def plot_bars(ax):
        # 1. DEEPSPEED
        ax.bar(0, ds_compute, width=bar_width, color=C_COMPUTE, edgecolor='black', linewidth=1, zorder=3)
        ax.bar(0, ds_transfer, bottom=ds_compute, width=bar_width, 
               color=C_TRANSFER, edgecolor='black', linewidth=1, hatch='///', zorder=3)

        # 2. DJINN
        ax.bar(1, dj_compute, width=bar_width, color=C_COMPUTE, edgecolor='black', linewidth=1, zorder=3)
        ax.bar(1, dj_overhead, bottom=dj_compute, width=bar_width, 
               color=C_OVERHEAD, edgecolor='black', linewidth=1, zorder=3)
        ax.bar(1, dj_stream_duration, bottom=0, width=bar_width, 
               color=C_STREAM, edgecolor='black', linewidth=1, hatch='...', alpha=0.9, zorder=4)

    plot_bars(ax1)
    plot_bars(ax2)

    # --- AXIS LIMITS ---
    ax1.set_ylim(32, 42)
    ax2.set_ylim(0, 2.8)

    # --- AXIS STYLING ---
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # Cut lines
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

    # --- ANNOTATIONS ---
    
    # 1. DeepSpeed Label (Straight Arrow)
    # Aligned perfectly vertically (x=0 to x=0)
    ax1.annotate(f"Full Reload\n{ds_data_gb} GB", 
                 xy=(0, ds_total), xycoords='data',
                 xytext=(0, 40), textcoords='data',
                 arrowprops=dict(arrowstyle="->", color=C_TRANSFER, lw=1.5), # No connectionstyle = Straight
                 ha='center', va='center', fontsize=11, fontweight='bold', color=C_TRANSFER)
    
    # 2. Speedup Label
    speedup = ds_total / dj_total
    ax2.text(1, 2.45, f"{speedup:.1f}× Speedup", 
             ha='center', va='center', fontsize=13, fontweight='bold', color=C_TEXT,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=1))

    # 3. Djinn Label (Curved Arrow)
    # rad=-0.45 gives it just enough loop to clear the overhead bar
    ax2.annotate(f"Delta Stream\n{dj_data_gb} GB", 
                 xy=(1 + bar_width/2, dj_stream_duration/2), xycoords='data', 
                 xytext=(1, 1.7), textcoords='data',
                 arrowprops=dict(arrowstyle="->", color=C_COMPUTE, lw=1.5, 
                                 connectionstyle="arc3,rad=-0.45"), 
                 ha='center', va='center', fontsize=11, fontweight='bold', color=C_COMPUTE)

    # --- LEGEND ---
    legend_elements = [
        mpatches.Patch(facecolor=C_TRANSFER, hatch='///', edgecolor='black', label='Blocking Transfer'),
        mpatches.Patch(facecolor=C_COMPUTE, edgecolor='black', label='GPU Compute'),
        mpatches.Patch(facecolor=C_STREAM, hatch='...', edgecolor='black', label='Async Stream (Overlap)'),
        mpatches.Patch(facecolor=C_OVERHEAD, edgecolor='black', label='System Overhead'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=10)

    # --- LABELS ---
    fig.text(0.02, 0.5, 'Time-to-First-Token (s)', va='center', rotation='vertical', fontsize=14, color=C_TEXT, fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([
        "DeepSpeed (Baseline)\nSynchronous", 
        "Djinn (Ours)\nAsynchronous"
    ], fontweight='bold', color=C_TEXT)

    ax1.yaxis.grid(True, linestyle=':', which='major', color='gray', alpha=0.5)
    ax2.yaxis.grid(True, linestyle=':', which='major', color='gray', alpha=0.5)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    # --- FINALIZE LAYOUT ---
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, hspace=0.15) 

    # --- DRAW THE GAP BAND ---
    bbox1 = ax1.get_position()
    bbox2 = ax2.get_position()
    gap_bottom = bbox2.y1
    gap_top = bbox1.y0
    gap_height = gap_top - gap_bottom
    
    gap_rect = mpatches.Rectangle(
        (bbox2.x0, gap_bottom),    
        bbox2.width,               
        gap_height,                
        transform=fig.transFigure, 
        facecolor='#e0e0e0',       
        edgecolor='none',          
        hatch='\\\\',              
        alpha=0.5,
        zorder=0                   
    )
    fig.patches.append(gap_rect)
    
    filename = 'exp2_memory.pdf'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Generated {filename}")

if __name__ == "__main__":
    plot_osdi_figure()