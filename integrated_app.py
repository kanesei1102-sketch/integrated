import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot

# ---------------------------------------------------------
# Safe Import of Optional Libraries
# ---------------------------------------------------------
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

# ---------------------------------------------------------
# 0. Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat V14 (日本語版)", layout="wide")

# ---------------------------------------------------------
# 1. Common Logic Functions
# ---------------------------------------------------------

def parse_vals(text):
    if not text: return []
    # Replace comma with newline and handle full-width numbers just in case
    text = text.replace(',', '\n').translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    vals = []
    for x in text.split('\n'):
        x = x.strip()
        if x:
            try:
                v = float(x)
                if not np.isnan(v): vals.append(v)
            except ValueError: pass
    return vals

def clean_data_for_log(vals):
    arr = np.array(vals)
    positive = arr[arr > 0]
    if len(positive) < len(arr):
        return positive.tolist(), True
    return positive.tolist(), False

def check_data_validity(values_list):
    if not values_list: return False
    return all(len(v) >= 2 for v in values_list)

def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def run_fallback_posthoc(groups_vals, group_names):
    sig_pairs = []
    n_groups = len(groups_vals)
    n_pairs = (n_groups * (n_groups - 1)) / 2
    if n_pairs == 0: return []
    
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            try:
                _, p = stats.mannwhitneyu(groups_vals[i], groups_vals[j], alternative='two-sided')
                p_adj = p * n_pairs # Bonferroni correction
                if p_adj < 0.05:
                    sig_pairs.append({'g1': group_names[i], 'g2': group_names[j], 'label': get_sig_label(p_adj)})
            except: pass
    return sig_pairs

def auto_select_test(groups_vals):
    """
    Automatic Statistical Test Selection Logic
    Returns: p_val, method_name, is_parametric, context_dict
    """
    context = {
        "small_n": False,
        "all_normal": True,
        "is_equal_var": True,
        "posthoc": "None"
    }

    if not check_data_validity(groups_vals):
        return 1.0, "データ不足", False, context

    # 1. Check Sample Size
    if any(len(v) < 3 for v in groups_vals):
        context["small_n"] = True

    # 2. Check Normality (Shapiro-Wilk)
    for v in groups_vals:
        if len(v) >= 3:
            if stats.shapiro(v)[1] <= 0.05: context["all_normal"] = False
    
    # 3. Check Homogeneity of Variance (Levene)
    try: _, p_lev = stats.levene(*groups_vals); context["is_equal_var"] = (p_lev > 0.05)
    except: context["is_equal_var"] = True

    method_name = ""
    p_val = 1.0

    if len(groups_vals) == 2:
        context["posthoc"] = "-"
        if context["all_normal"]:
            if context["is_equal_var"]:
                method_name = "Studentのt検定"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=True)
            else:
                method_name = "Welchのt検定"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=False)
        else:
            method_name = "Mann-WhitneyのU検定"
            _, p_val = stats.mannwhitneyu(groups_vals[0], groups_vals[1], alternative='two-sided')
    else:
        if context["all_normal"] and context["is_equal_var"]:
            method_name = "一元配置分散分析 (ANOVA)"
            context["posthoc"] = "Tukey-Kramer検定"
            _, p_val = stats.f_oneway(*groups_vals)
        else:
            method_name = "Kruskal-Wallis検定"
            context["posthoc"] = "Dunn検定 (Bonferroni)" if HAS_POSTHOCS else "Mann-Whitney U (Bonferroni)"
            _, p_val = stats.kruskal(*groups_vals)

    return p_val, method_name, context["all_normal"], context

def calculate_sig_bars_layout(pairs, name_to_x, base_y_map, step_y, is_log):
    bars_to_draw = []
    levels = {}

    for p in pairs:
        g1, g2, label = p['g1'], p['g2'], p['label']
        if g1 not in name_to_x or g2 not in name_to_x: continue
        
        x1 = min(name_to_x[g1], name_to_x[g2])
        x2 = max(name_to_x[g1], name_to_x[g2])
        
        y_start_1 = base_y_map.get(g1, 0)
        y_start_2 = base_y_map.get(g2, 0)
        current_base_y = max(y_start_1, y_start_2)
        
        lvl = 0
        while True:
            collision = False
            for (occ_x1, occ_x2, _) in levels.get(lvl, []):
                if not (x2 < occ_x1 - 0.1 or x1 > occ_x2 + 0.1): 
                    collision = True
                    break
            if not collision: break
            lvl += 1
        
        if is_log:
            bar_y = current_base_y * (1.15 ** (lvl + 1))
        else:
            bar_y = current_base_y + (step_y * (lvl + 1))
            
        if lvl not in levels: levels[lvl] = []
        levels[lvl].append((x1, x2, bar_y))
        
        bars_to_draw.append({'x1': x1, 'x2': x2, 'y': bar_y, 'label': label})
        
    return bars_to_draw

# ---------------------------------------------------------
# 2. Plotting Functions (Matplotlib)
# ---------------------------------------------------------

def draw_matplotlib_1factor(data_dict, sig_pairs, config, is_norm):
    plt.rcParams['font.family'] = 'sans-serif'
    group_names = list(data_dict.keys())
    all_values = list(data_dict.values())
    
    # X-axis positioning
    x_pos = np.arange(len(group_names)) * config['spacing']
    name_to_x = {name: x for name, x in zip(group_names, x_pos)}

    # Figure Width adjustment
    base_width_per_group = 3.0 
    fig_w = max(6.0, len(data_dict) * base_width_per_group * config['spacing'])
    
    fig, ax = plt.subplots(figsize=(fig_w, config['height']))
    # --- TICK DIRECTION FIX: POINT INWARD ---
    ax.tick_params(direction='in', top=False, right=False, which='both')
    
    all_flat = [x for sub in all_values for x in sub]
    max_v = max(all_flat) if all_flat else 1
    pos_vals = [x for x in all_flat if x > 0]
    min_pos_v = min(pos_vals) if pos_vals else 0.01
    
    base_y_map = {} 
    
    # Determine plot type based on logic
    final_type = config['manual_type'] if config['mode'].startswith("手動") else ("箱ひげ図 (Box)" if not is_norm else "棒グラフ (Bar)")

    for i, (name, vals) in enumerate(data_dict.items()):
        vals = np.array(vals); p = x_pos[i]
        
        if config['scale'].startswith("Log"):
            vals_plot, _ = clean_data_for_log(vals)
            vals_plot = np.array(vals_plot)
        else:
            vals_plot = vals

        mean = np.mean(vals_plot) if len(vals_plot)>0 else 0
        std = np.std(vals_plot, ddof=1) if len(vals_plot)>1 else 0
        sem = std/np.sqrt(len(vals_plot)) if len(vals_plot)>0 else 0
        err = sem if config['error'].startswith("SEM") else std
        col = config['colors'].get(name, "#333333")
        
        top_val = max(vals_plot) if len(vals_plot)>0 else 0
        if "Bar" in final_type:
            top_val = mean + (err if config['error'] != "なし" else 0)
        
        margin_ratio = 1.05 if config['scale'].startswith("Linear") else 1.2
        base_y_map[name] = top_val * margin_ratio
        
        # Draw Graphs
        if "Bar" in final_type:
            ax.bar(p, mean, width=config['bar_width'], color=col, edgecolor='black', alpha=0.8, zorder=1)
            if config['error'] != "なし":
                ax.errorbar(p, mean, yerr=err, fmt='none', c='black', capsize=5, zorder=2)
        elif "Box" in final_type and len(vals_plot)>0:
            ax.boxplot(vals_plot, positions=[p], widths=config['bar_width'], patch_artist=True, 
                       boxprops=dict(facecolor=col, alpha=0.8), medianprops=dict(color='black'), showfliers=False, zorder=1)
        elif "Violin" in final_type and len(vals_plot)>0:
            parts = ax.violinplot(vals_plot, positions=[p], widths=config['bar_width'], showextrema=False)
            for pc in parts['bodies']: pc.set_facecolor(col); pc.set_alpha(0.8); pc.set_zorder(1)
            
        # Jitter
        if len(vals_plot) > 0:
            jitter_width = config['bar_width'] * 0.4 * config['jitter'] 
            noise = np.random.uniform(-1, 1, len(vals_plot)) * jitter_width
            ax.scatter(p + noise, vals_plot, s=config['dot_size'], facecolors='white', edgecolors='#555555', zorder=3, alpha=config['dot_alpha'])

    # Sig Bars
    step_y = max_v * 0.1
    is_log = config['scale'].startswith("Log")
    if is_log: ax.set_yscale('log')
    
    bars = calculate_sig_bars_layout(sig_pairs, name_to_x, base_y_map, step_y, is_log)
    
    global_max_y = max_v
    for b in bars:
        x1, x2, y, label = b['x1'], b['x2'], b['y'], b['label']
        ax.plot([x1, x1, x2, x2], [y*0.98, y, y, y*0.98], lw=1.5, c='black')
        ax.text((x1+x2)/2, y, label, ha='center', va='bottom', fontsize=12)
        if y > global_max_y: global_max_y = y

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_names, fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.set_title(config['title'], fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Fix X-axis limits
    min_x = min(x_pos) - 1.0
    max_x = max(x_pos) + 1.0
    ax.set_xlim(min_x, max_x)
    
    if config['manual_y_max'] > 0:
        ax.set_ylim(bottom=None, top=config['manual_y_max'])
    else:
        top_margin = 1.1 if not is_log else 1.5
        bottom_val = 0 if not is_log else min_pos_v * 0.5
        if config['scale'].startswith("Linear") and config['auto_zoom']:
             ax.set_ylim(bottom=0, top=global_max_y * top_margin)
        else:
             ax.set_ylim(bottom=bottom_val, top=global_max_y * top_margin)

    return fig

def draw_matplotlib_2factor(df_raw, grouped_data, sig_res_map, config, sub_names):
    plt.rcParams['font.family'] = 'sans-serif'
    n_major = len(grouped_data)
    n_sub = len(sub_names)
    
    x_base = np.arange(n_major) * config['spacing']
    
    base_width_per_major = max(4.0, n_sub * 1.5)
    fig_w = max(6.0, n_major * base_width_per_major * config['spacing'])
    
    fig, ax = plt.subplots(figsize=(fig_w, config['height']))
    # --- TICK DIRECTION FIX: POINT INWARD ---
    ax.tick_params(direction='in', top=False, right=False, which='both')
    
    w = config['bar_width']
    
    total_group_width = w * n_sub * 1.2
    offsets = np.linspace(-total_group_width/2 + w/2 + (w*0.1), total_group_width/2 - w/2 - (w*0.1), n_sub)
    
    all_raw = df_raw['Val'].tolist()
    max_v = max(all_raw) if all_raw else 1
    pos_vals = [x for x in all_raw if x > 0]
    min_pos_v = min(pos_vals) if pos_vals else 0.01
    
    name_to_x_map = {}
    base_y_map = {} 
    is_log = config['scale'].startswith("Log")
    if is_log: ax.set_yscale('log')
    
    for i, s_name in enumerate(sub_names):
        col = config['colors'].get(s_name, "#333333")
        means, errs, raw_vals_list = [], [], []
        x_coords = x_base + offsets[i]
        
        for j, m_group in enumerate(grouped_data.keys()):
            v = grouped_data[m_group].get(s_name, [])
            if is_log: v, _ = clean_data_for_log(v)
            else: v = v if isinstance(v, list) else []
            name_to_x_map[(m_group, s_name)] = x_coords[j]
            
            if len(v) > 0:
                mean = np.mean(v)
                std = np.std(v, ddof=1) if len(v)>1 else 0
                sem = std/np.sqrt(len(v)) if len(v)>0 else 0
                err = sem if config['error'].startswith("SEM") else std
                means.append(mean); errs.append(err)
            else:
                means.append(0); errs.append(0)
            raw_vals_list.append(v)
            
            top = max(v) if len(v)>0 else 0
            if "Bar" in config['manual_type']: top = (means[-1] + errs[-1]) if len(v)>0 else 0
            margin = 1.2 if is_log else 1.05
            base_y_map[(m_group, s_name)] = top * margin

        if "Bar" in config['manual_type']: 
            ax.bar(x_coords, means, width=w, label=s_name, color=col, edgecolor='black', alpha=0.8, yerr=errs, capsize=4, zorder=1)
        else:
            for k, v in enumerate(raw_vals_list):
                if len(v) > 0:
                    ax.boxplot(v, positions=[x_coords[k]], widths=w*0.8, patch_artist=True, 
                               boxprops=dict(facecolor=col, alpha=0.8), medianprops=dict(color='black'), showfliers=False, zorder=1)

        for k, v in enumerate(raw_vals_list):
            if len(v) > 0:
                jitter_width = w * 0.4 * config['jitter']
                noise = np.random.uniform(-1, 1, len(v)) * jitter_width
                ax.scatter(x_coords[k] + noise, v, s=config['dot_size'], facecolors='white', edgecolors='#555555', zorder=3, alpha=config['dot_alpha'])

    global_max_y = max_v
    step_y = max_v * 0.1
    
    for m_group in grouped_data.keys():
        pairs = sig_res_map.get(m_group, [])
        if not pairs: continue
        
        local_name_to_x = {s: name_to_x_map[(m_group, s)] for s in sub_names}
        local_base_y = {s: base_y_map[(m_group, s)] for s in sub_names}
        
        bars = calculate_sig_bars_layout(pairs, local_name_to_x, local_base_y, step_y, is_log)
        
        for b in bars:
            x1, x2, y, label = b['x1'], b['x2'], b['y'], b['label']
            ax.plot([x1, x1, x2, x2], [y*0.98, y, y, y*0.98], lw=1.5, c='black')
            ax.text((x1+x2)/2, y, label, ha='center', va='bottom', fontsize=12)
            if y > global_max_y: global_max_y = y

    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.set_title(config['title'], fontsize=14)
    ax.set_xticks(x_base)
    ax.set_xticklabels(list(grouped_data.keys()), fontsize=12)
    
    if "Bar" in config['manual_type']:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=config['colors'].get(n,'#333'), edgecolor='black', label=n) for n in sub_names]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    min_x = min(x_base) - 0.8
    max_x = max(x_base) + 0.8
    ax.set_xlim(min_x, max_x)
    
    if config['manual_y_max'] > 0:
        ax.set_ylim(bottom=None, top=config['manual_y_max'])
    else:
        top_margin = 1.1 if not is_log else 1.5
        bottom_val = 0 if not is_log else min_pos_v * 0.5
        ax.set_ylim(bottom=bottom_val, top=global_max_y * top_margin)

    return fig

# ---------------------------------------------------------
# 2. Sidebar Settings (Japanese)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### [重要：論文等での使用について]")
    st.warning("""
    **研究成果として論文発表等を予定されていますか？**
    本ツールはベータ版です。学術目的で使用される場合は、**事前に開発者（金子）までご連絡ください。**
    共著者への追加や謝辞（Acknowledgments）への記載について相談させていただきます。
    👉 **[お問い合わせフォーム](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)
    st.divider()

    analysis_mode = st.radio("解析モード", ["1要因 (単純比較: A vs B vs C)", "2要因 (二元配置分散分析: 要因A × 要因B)"], 
                             help="1要因: 1つの条件で群間比較\n2要因: 2つの条件の掛け合わせ")
    st.divider()

    st.header("🛠️ グラフ設定")
    with st.expander("📈 種類とスケール", expanded=True):
        if analysis_mode.startswith("1要因"):
            graph_mode_ui = st.radio("選択モード", ["自動 (推奨)", "手動"])
            scale_option = st.radio("Y軸スケール", ["Linear (線形)", "Log (対数)"])
            auto_zoom = st.checkbox("外れ値を除外して拡大 (Zoom)", value=False) if scale_option.startswith("Linear") else False
            
            manual_graph_type = "棒グラフ (Bar)"
            error_type = "SD (標準偏差)"
            if graph_mode_ui.startswith("手動"):
                manual_graph_type = st.selectbox("グラフの種類", ["棒グラフ (Bar)", "箱ひげ図 (Box)", "バイオリン図 (Violin)"])
                if "Bar" in manual_graph_type:
                    error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"])
                else: error_type = "なし"
            else:
                st.caption("※ データの分布に基づいて自動選択されます")
                error_type = "SD (標準偏差)"
        else:
            graph_type_2way = st.selectbox("グラフの種類", ["棒グラフ (Bar)", "箱ひげ図 (Box)"])
            error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"]) if "Bar" in graph_type_2way else "なし"
            scale_option = st.radio("Y軸スケール", ["Linear (線形)", "Log (対数)"])
            graph_mode_ui = "手動"; manual_graph_type = graph_type_2way; auto_zoom = False

    with st.expander("🎨 デザイン微調整", expanded=False):
        fig_title = st.text_input("グラフタイトル", value="実験結果")
        y_axis_label = st.text_input("Y軸ラベル", value="相対値")
        manual_y_max = st.number_input("Y軸最大値 (0で自動)", value=0.0, step=1.0)
        st.divider()
        fig_height = st.slider("画像の高さ", 3.0, 15.0, 6.0)
        bar_width = st.slider("バーの太さ", 0.1, 1.0, 0.35, 0.05)
        
        label_spacing = "グループ間隔 (1.0=最大)" if analysis_mode.startswith("1要因") else "要因間隔 (1.0=最大)"
        group_spacing = st.slider(label_spacing, 0.2, 1.0, 1.0, 0.05)
        
        st.caption("ドット・その他")
        dot_size = st.slider("ドットサイズ", 0, 20, 6)
        dot_alpha = st.slider("ドットの透明度", 0.1, 1.0, 0.7)
        jitter = st.slider("散らばり (Jitter)", 0.0, 1.0, 0.2)

# ---------------------------------------------------------
# 3. Main Area: Data Input
# ---------------------------------------------------------
st.title("🔬 Ultimate Sci-Stat & Graph Engine V14 (日本語版)")

plot_config = {
    'mode': graph_mode_ui, 'manual_type': manual_graph_type, 'scale': scale_option,
    'error': error_type, 'auto_zoom': auto_zoom, 'title': fig_title, 'ylabel': y_axis_label,
    'width': 0, 'height': fig_height, 'bar_width': bar_width, 'spacing': group_spacing,
    'dot_size': dot_size, 'dot_alpha': dot_alpha, 'jitter': jitter, 'colors': {}, 'manual_y_max': manual_y_max
}

data_dict = {}
grouped_data = {}

# === 1-Factor Input ===
if analysis_mode.startswith("1要因"):
    st.caption("単一条件下での複数群比較を行います。")
    t1, t2 = st.tabs(["✍️ 手動入力", "📂 CSVアップロード"])
    
    
    if 'csv_data_cache' not in st.session_state:
        st.session_state.csv_data_cache = {}

    with t1:
        if 'g_cnt' not in st.session_state: st.session_state.g_cnt = 3
        c1, c2 = st.columns([1,5])
        if c1.button("＋"): st.session_state.g_cnt += 1
        if c2.button("－"): st.session_state.g_cnt = max(2, st.session_state.g_cnt - 1)
        cols = st.columns(min(st.session_state.g_cnt, 4))
        for i in range(st.session_state.g_cnt):
            with cols[i%4]:
                name = st.text_input(f"グループ {i+1}", f"Group {i+1}", key=f"n{i}")
                raw = st.text_area(f"数値 {i+1}", key=f"d{i}")
                v = parse_vals(raw); 
                if v: data_dict[name] = v
                
    with t2:
        up = st.file_uploader("CSVファイル", type="csv")
        if up:
            try:
                df = pd.read_csv(up)
                st.write("プレビュー:", df.head(3))
                
                
                fmt = st.radio("データ形式", ["ロング形式 (縦持ち: Group列とValue列)", "ワイド形式 (横持ち: 各列がグループ)"])
                
                if fmt.startswith("ロング"):
                    cols = df.columns.tolist()
                    c_grp = st.selectbox("グループ列", cols)
                    c_val = st.selectbox("数値列", [c for c in cols if c!=c_grp])
                    
                    
                    if st.button("データを読み込む"):
                        temp_data = {}
                        for g in df[c_grp].unique():
                            v = df[df[c_grp]==g][c_val].dropna().tolist()
                            clean = [float(x) for x in v if str(x).replace('.','').isdigit()]
                            if clean: temp_data[g] = clean
                        st.session_state.csv_data_cache = temp_data # 保存
                        
                else:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    sel = st.multiselect("列を選択", num_cols, default=list(num_cols)[:3])
                    
                    
                    if st.button("データを読み込む"):
                        temp_data = {}
                        for c in sel:
                            v = df[c].dropna().tolist(); 
                            if v: temp_data[c] = v
                        st.session_state.csv_data_cache = temp_data 

                
                if st.session_state.csv_data_cache:
                    st.success("CSVからデータを読み込みました！")
                    data_dict.update(st.session_state.csv_data_cache)
                    
                    
                    if st.button("CSVデータをクリア"):
                        st.session_state.csv_data_cache = {}
                        st.rerun()
                        
            except Exception as e: st.error(str(e))

# === 2-Factor Input ===
else:
    st.caption("2要因の交互作用分析 (要因A × 要因B)")
    c1, c2 = st.columns(2)
    with c1:
        mj_str = st.text_area("要因A (X軸) *改行区切り", "DMSO\nDrug_X\nDrug_Y", height=100)
        mj_grps = [x.strip() for x in mj_str.split('\n') if x.strip()]
    with c2:
        if 'sub_cnt' not in st.session_state: st.session_state.sub_cnt = 2
        sc1, sc2 = st.columns(2)
        if sc1.button("＋サブグループ"): st.session_state.sub_cnt += 1
        if sc2.button("－削除"): st.session_state.sub_cnt = max(2, st.session_state.sub_cnt - 1)
        sub_names = []
        for i in range(st.session_state.sub_cnt):
            sub_names.append(st.text_input(f"Sub {i+1}", f"Sub {i+1}", key=f"s{i}"))
    st.divider()
    if mj_grps and sub_names:
        tabs = st.tabs(mj_grps)
        for i, m in enumerate(mj_grps):
            grouped_data[m] = {}
            with tabs[i]:
                cols = st.columns(len(sub_names))
                for j, s in enumerate(sub_names):
                    with cols[j]:
                        raw = st.text_area(f"{s}", key=f"d2_{i}_{j}")
                        v = parse_vals(raw); 
                        if v: grouped_data[m][s] = v

# ---------------------------------------------------------
# 4. Color Settings
# ---------------------------------------------------------
with st.sidebar:
    with st.expander("🖍️ 配色設定", expanded=True):
        defs = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
        if analysis_mode.startswith("1要因") and data_dict:
            for i, k in enumerate(data_dict.keys()):
                plot_config['colors'][k] = st.color_picker(k, defs[i%len(defs)])
        elif analysis_mode.startswith("2要因") and 'sub_names' in locals():
            for i, k in enumerate(sub_names):
                plot_config['colors'][k] = st.color_picker(k, defs[i%len(defs)])

# ---------------------------------------------------------
# 5. Execution (Report & Draw)
# ---------------------------------------------------------
if analysis_mode.startswith("1要因"):
    if len(data_dict) >= 2 and check_data_validity(data_dict.values()):
        # Calc & Context
        p_val, method, is_norm, ctx = auto_select_test(list(data_dict.values()))
        st.success(f"解析完了: {method}")
        
        # --- Report Logic ---
        easy_reason = ""
        if ctx["all_normal"] and ctx["is_equal_var"]:
            easy_reason = "データの分布に著しい偏りがなく、等分散性が棄却されなかったため、統計的検定力の高い標準的な「パラメトリック検定」が選択されました。"
        elif not ctx["all_normal"]:
            easy_reason = "データの分布に正規性の欠如、または外れ値の可能性が示唆されたため、順位に基づく「ノンパラメトリック検定」が選択されました。"
        else:
            easy_reason = "正規性は棄却されませんでしたが、等分散性が棄却されました。そのため、分散が等しくない場合でも頑健な手法が選択されました。"

        if ctx["small_n"]:
            easy_reason += "\n   ※ サンプルサイズが小さい群が含まれるため、分布の判定は限定的です。"

        result_summary = "【有意差あり】" if p_val < 0.05 else "【有意差なし】"
        conclusion_text = "本データセットにおいて群間に統計的に有意な差が認められました。少なくとも一部の群間で平均値（または中央値）が異なることが示唆されます。" if p_val < 0.05 else "本データセットにおいて群間に統計的に有意な差は認められませんでした。平均値の明確な違いは確認できません。"

        norm_res_text = "有意な偏りなし (棄却されず)" if ctx["all_normal"] else "非正規性が示唆される (棄却)"
        if ctx["small_n"]: norm_res_text += " *参考値 (n<3)"
        var_res_text = "等分散性は棄却されず (Not Rejected)" if ctx["is_equal_var"] else "等分散性は棄却された (Rejected)"

        analysis_path = f"""
【統計手法の選定プロセス (自動診断)】
1. 正規性検定 (Shapiro-Wilk): {norm_res_text}
2. 等分散性検定 (Levene): {var_res_text}
⇒ 上記の診断に基づき、**{method}** が採用されました。
"""
        
        with st.expander("📝 そのまま使えるレポート案 (詳細)", expanded=True):
            full_report = f"""
【解析レポート: {", ".join(data_dict.keys())} の比較】{analysis_path}

1. 検定手法の選定理由:
   手法: {method}
   理由: {easy_reason}

2. 解析結果:
   判定: {result_summary}
   全体のP値: {p_val:.4e}
   (有意水準 α=0.05)

3. 事後検定 (多重比較):
   {"多重比較検定が行われ、有意差はグラフに反映されています。" if len(data_dict) > 2 else "2群間の直接比較が行われました。"}

4. 結論:
   {conclusion_text}
            """
            st.text_area("レポート全文", value=full_report, height=400)
            
        with st.expander("📄 論文用 'Methods' セクション案 (英語/日本語)", expanded=False):
            methods_text = f"""
[Japanese]
統計解析にはPythonおよびSciPyライブラリを使用した。
データの正規性はShapiro-Wilk検定、等分散性はLevene検定により評価した。
群間比較には{method}を用いた。
{f"(事後検定: {ctx['posthoc']})" if len(data_dict) > 2 else ""}
P値 0.05未満を統計的有意差ありと判定した。

[English]
Statistical analyses were performed using Python with the SciPy library.
Normality of data was assessed using the Shapiro-Wilk test, and homogeneity of variance was assessed using Levene's test.
Comparisons between groups were determined using {method}.
{f"(Post-hoc analysis: {ctx['posthoc']})" if len(data_dict) > 2 else ""}
A P-value of less than 0.05 was considered statistically significant.
            """
            st.text_area("Methods案", value=methods_text, height=250)

        # Posthoc
        sig_pairs = []
        grps = list(data_dict.keys()); vals = list(data_dict.values())
        if p_val < 0.05:
            if len(data_dict)==2:
                sig_pairs.append({'g1': grps[0], 'g2': grps[1], 'label': get_sig_label(p_val)})
            elif "ANOVA" in method:
                flat_d = [x for sub in vals for x in sub]
                flat_l = [n for n, sub in data_dict.items() for _ in sub]
                tuk = pairwise_tukeyhsd(flat_d, flat_l)
                for _, r in pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0]).iterrows():
                    if r['reject']: sig_pairs.append({'g1': r['group1'], 'g2': r['group2'], 'label': get_sig_label(r['p-adj'])})
            elif HAS_POSTHOCS:
                dunn = sp.posthoc_dunn(vals, p_adjust='bonferroni')
                dunn.columns = grps; dunn.index = grps
                for i in range(len(grps)):
                    for j in range(i+1, len(grps)):
                        if dunn.iloc[i, j] < 0.05:
                            sig_pairs.append({'g1': grps[i], 'g2': grps[j], 'label': get_sig_label(dunn.iloc[i, j])})
            else:
                st.warning("scikit-posthocsがインストールされていません。代替ロジック(Bonferroni-MannWhitney)を実行します。")
                sig_pairs = run_fallback_posthoc(vals, grps)
        
        # Draw (Matplotlib)
        try:
            fig = draw_matplotlib_1factor(data_dict, sig_pairs, plot_config, is_norm)
            st.pyplot(fig)
            buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            st.download_button("📥 画像を保存 (PNG)", buf, file_name="result.png", mime="image/png")
        except Exception as e: st.error(f"Plot Error: {e}")
    else: st.info("データを入力してください。")

else: # 2-Factor
    if len(grouped_data) > 0:
        rows = []
        for m, sub in grouped_data.items():
            for s, v in sub.items():
                for x in v: rows.append({'A': m, 'B': s, 'Val': x})
        df_a = pd.DataFrame(rows)
        
if not df_a.empty:
            try:
                # 統計モデルの構築と計算
                model = ols('Val ~ C(A) * C(B)', data=df_a).fit()
                res = sm.stats.anova_lm(model, typ=2)
                p_int = res.loc['C(A):C(B)', 'PR(>F)']
                p_a = res.loc['C(A)', 'PR(>F)']
                p_b = res.loc['C(B)', 'PR(>F)']
                
                # 前提条件の診断用計算
                res_norm = stats.shapiro(model.resid)[1]
                all_groups_vals = [df_a[(df_a['A']==m) & (df_a['B']==s)]['Val'].tolist() for m in mj_grps for s in sub_names]
                res_levene = stats.levene(*[v for v in all_groups_vals if len(v)>=2])[1]
                
                # --- ここから詳細レポート表示 ---
                with st.expander("📝 二元配置分散分析 詳細レポート案", expanded=True):
                    st.markdown(f"""
【二元配置分散分析 (Two-way ANOVA) 詳細レポート】
対象要因: A ({", ".join(mj_grps)}) × B ({", ".join(sub_names)})

1. 前提条件の診断:
   - 正規性検定 (残差のShapiro-Wilk): P={res_norm:.4f} ({"正規性を維持" if res_norm > 0.05 else "非正規性の疑いあり"})
   - 等分散性検定 (Levene): P={res_levene:.4f} ({"等分散性は維持" if res_levene > 0.05 else "不等分散の疑いあり"})
   ⇒ 二元配置分散分析を実行しました。

2. 解析結果サマリー:
   - 要因A 主効果: P = {p_a:.4e} ({"有意差あり" if p_a < 0.05 else "有意差なし"})
   - 要因B 主効果: P = {p_b:.4e} ({"有意差あり" if p_b < 0.05 else "有意差なし"})
   - 交互作用 (A×B): P = {p_int:.4e} ({"有意差あり" if p_int < 0.05 else "有意差なし"})

3. 結論と解釈:
   {"交互作用が有意であるため、要因Aの効果は要因Bの状態によって変化します。" if p_int < 0.05 else "交互作用は有意ではなく、各要因は独立して影響を及ぼしています。"}
   {"このため、各水準における「単純主効果（層別解析）」を優先して解釈する必要があります。" if p_int < 0.05 else "各主効果の判定に基づいた結論を導出してください。"}
                    """)
                    
                    st.text_area("Methods案 (2-way)", f"[English] Two-way analysis of variance (ANOVA) was used to assess the interaction between {mj_grps} and {sub_names}. Significant interactions were further analyzed for simple main effects. P < 0.05 was considered significant.\n\n[Japanese] {mj_grps}と{sub_names}の交互作用を評価するため二元配置分散分析を用いた。交互作用が有意な場合は単純主効果を検定した。P<0.05を有意とした。", height=150)
                # --- レポート表示ここまで ---

                with st.expander("📊 分散分析表 (ANOVA Table)", expanded=False):
                    st.write(res)
                    st.info(f"交互作用 (Interaction): **{'あり' if p_int < 0.05 else 'なし'}** (P={p_int:.4f})")
                    fig_i, ax_i = plt.subplots()
                    interaction_plot(x=df_a['A'], trace=df_a['B'], response=df_a['Val'], ax=ax_i)
                    st.pyplot(fig_i)
            except: st.warning("ANOVAの計算に失敗しました。データを確認してください。")

            st.subheader("単純主効果 (層別解析)")
            sig_res_map = {}
            report_text = ""
            
            for m, sub in grouped_data.items():
                s_keys = list(sub.keys()); s_vals = list(sub.values())
                if not check_data_validity(s_vals): continue
                
                p, method, _, _ = auto_select_test(s_vals)
                report_text += f"- **{m}**: P={p:.4f} ({method})\n"
                
                sig_res_map[m] = []
                if p < 0.05:
                    if len(s_vals) == 2:
                        sig_res_map[m].append({'g1': s_keys[0], 'g2': s_keys[1], 'label': get_sig_label(p)})
                    elif "ANOVA" in method:
                        flat_d = [x for sub in s_vals for x in sub]
                        flat_l = [n for n, sub in zip(s_keys, s_vals) for _ in sub]
                        tuk = pairwise_tukeyhsd(flat_d, flat_l)
                        for _, r in pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0]).iterrows():
                            if r['reject']: sig_res_map[m].append({'g1': r['group1'], 'g2': r['group2'], 'label': get_sig_label(r['p-adj'])})
                    else:
                        if HAS_POSTHOCS:
                            dunn = sp.posthoc_dunn(s_vals, p_adjust='bonferroni')
                            dunn.columns = s_keys; dunn.index = s_keys
                            for i in range(len(s_keys)):
                                for j in range(i+1, len(s_keys)):
                                    if dunn.iloc[i, j] < 0.05:
                                        sig_res_map[m].append({'g1': s_keys[i], 'g2': s_keys[j], 'label': get_sig_label(dunn.iloc[i, j])})
                        else:
                            sig_res_map[m] = run_fallback_posthoc(s_vals, s_keys)
            
            st.markdown(report_text)

            # Draw (Matplotlib)
            try:
                fig = draw_matplotlib_2factor(df_a, grouped_data, sig_res_map, plot_config, sub_names)
                st.pyplot(fig)
                buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                st.download_button("📥 画像を保存 (PNG)", buf, file_name="result_2way.png", mime="image/png")
            except Exception as e: st.error(f"Plot Error: {e}")
    else: st.info("データを入力してください。")

# ---------------------------------------------------------
# 6. Sidebar Footer: Disclaimer (Japanese)
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    st.caption("【免責事項】")
    st.caption("""
    本ソフトウェアは研究用として現状有姿で提供されるものです。
    開発者は、計算結果の正確性、完全性、特定目的への適合性について一切の保証を行いません。
    本ツールの使用から生じるいかなる損害（研究データの損失、論文の修正・撤回、機会損失等を含む）についても、開発者は責任を負いません。
    統計的妥当性の最終判断は、利用者の責任において行ってください。
    """)
