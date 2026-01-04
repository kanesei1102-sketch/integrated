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
# ライブラリの安全なインポート
# ---------------------------------------------------------
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

# ---------------------------------------------------------
# 0. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat V13 (Matplotlib)", layout="wide")

# ---------------------------------------------------------
# 1. 共通関数 (Logic)
# ---------------------------------------------------------

def parse_vals(text):
    """数値変換の厳密化"""
    if not text: return []
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
    """対数軸用に0以下を除外"""
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
                p_adj = p * n_pairs 
                if p_adj < 0.05:
                    sig_pairs.append({'g1': group_names[i], 'g2': group_names[j], 'label': get_sig_label(p_adj)})
            except: pass
    return sig_pairs

def auto_select_test(groups_vals):
    if not check_data_validity(groups_vals):
        return 1.0, "データ不足 (N<2)", False, "データ不足"

    all_normal = True
    for v in groups_vals:
        if len(v) >= 3:
            if stats.shapiro(v)[1] <= 0.05: all_normal = False
    
    try: _, p_lev = stats.levene(*groups_vals); is_equal_var = (p_lev > 0.05)
    except: is_equal_var = True

    method_name = ""
    p_val = 1.0
    reason = ""

    if len(groups_vals) == 2:
        if all_normal:
            if is_equal_var:
                method_name = "Student's t-test"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=True)
                reason = "正規分布かつ等分散"
            else:
                method_name = "Welch's t-test"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=False)
                reason = "正規分布だが不等分散"
        else:
            method_name = "Mann-Whitney U"
            _, p_val = stats.mannwhitneyu(groups_vals[0], groups_vals[1], alternative='two-sided')
            reason = "非正規分布 (または外れ値)"
    else:
        if all_normal and is_equal_var:
            method_name = "One-way ANOVA"
            _, p_val = stats.f_oneway(*groups_vals)
            reason = "正規分布かつ等分散"
        else:
            method_name = "Kruskal-Wallis"
            _, p_val = stats.kruskal(*groups_vals)
            reason = "非正規分布 (または不等分散)"

    return p_val, method_name, all_normal, reason

def calculate_sig_bars_layout(pairs, name_to_x, base_y_map, step_y, is_log):
    """Tetris Algorithm for Stacking"""
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
                # マージンを持たせて重なり判定
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
# 2. 描画関数 (Matplotlib Robust)
# ---------------------------------------------------------

def draw_matplotlib_1factor(data_dict, sig_pairs, config, is_norm):
    # 日本語フォント設定 (環境依存回避のため英語フォント推奨だが、文字化け対策でsans-serif)
    plt.rcParams['font.family'] = 'sans-serif'
    
    group_names = list(data_dict.keys())
    all_values = list(data_dict.values())
    
    # 幅計算
    fig_w = config['width'] if config['width'] > 0 else max(6.0, len(data_dict) * 1.5 * config['spacing'])
    fig, ax = plt.subplots(figsize=(fig_w, config['height']))
    
    x_pos = np.arange(len(group_names)) * config['spacing']
    name_to_x = {name: x for name, x in zip(group_names, x_pos)}
    
    all_flat = [x for sub in all_values for x in sub]
    max_v = max(all_flat) if all_flat else 1
    # Log scale safety
    pos_vals = [x for x in all_flat if x > 0]
    min_pos_v = min(pos_vals) if pos_vals else 0.01
    
    base_y_map = {} 
    final_type = config['manual_type'] if config['mode'].startswith("手動") else ("箱ひげ図 (Box)" if not is_norm else "棒グラフ (Bar)")

    # Plot Data
    for i, (name, vals) in enumerate(data_dict.items()):
        vals = np.array(vals); p = x_pos[i]
        
        # Log Safety
        if config['scale'] == "対数 (Log)":
            vals_plot, _ = clean_data_for_log(vals)
            vals_plot = np.array(vals_plot)
        else:
            vals_plot = vals

        mean = np.mean(vals_plot) if len(vals_plot)>0 else 0
        std = np.std(vals_plot, ddof=1) if len(vals_plot)>1 else 0
        sem = std/np.sqrt(len(vals_plot)) if len(vals_plot)>0 else 0
        err = sem if config['error'].startswith("SEM") else std
        col = config['colors'].get(name, "#333333")
        
        # Base Y
        top_val = max(vals_plot) if len(vals_plot)>0 else 0
        if "棒" in final_type:
            top_val = mean + (err if config['error'] != "None" else 0)
        
        margin_ratio = 1.05 if config['scale'].startswith("線形") else 1.2
        base_y_map[name] = top_val * margin_ratio
        
        if "棒" in final_type:
            ax.bar(p, mean, width=config['bar_width'], color=col, edgecolor='black', alpha=0.8, zorder=1)
            if config['error'] != "None":
                ax.errorbar(p, mean, yerr=err, fmt='none', c='black', capsize=5, zorder=2)
        elif "箱" in final_type and len(vals_plot)>0:
            ax.boxplot(vals_plot, positions=[p], widths=config['bar_width'], patch_artist=True, 
                       boxprops=dict(facecolor=col, alpha=0.8), medianprops=dict(color='black'), showfliers=False)
        elif "バイオリン" in final_type and len(vals_plot)>0:
            parts = ax.violinplot(vals_plot, positions=[p], widths=config['bar_width'], showextrema=False)
            for pc in parts['bodies']: pc.set_facecolor(col); pc.set_alpha(0.8)
            
        if len(vals_plot) > 0:
            noise = np.random.normal(0, config['jitter'], len(vals_plot))
            ax.scatter(p+noise, vals_plot, s=config['dot_size'], facecolors='white', edgecolors='#555555', zorder=3, alpha=config['dot_alpha'])

    # Sig Bars
    step_y = max_v * 0.1
    is_log = config['scale'] == "対数 (Log)"
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
    
    if config['manual_y_max'] > 0:
        ax.set_ylim(bottom=None, top=config['manual_y_max'])
    else:
        top_margin = 1.1 if not is_log else 1.5
        bottom_val = 0 if not is_log else min_pos_v * 0.5
        if config['scale'].startswith("線形") and config['auto_zoom']:
             ax.set_ylim(bottom=0, top=global_max_y * top_margin)
        else:
             ax.set_ylim(bottom=bottom_val, top=global_max_y * top_margin)

    return fig

def draw_matplotlib_2factor(df_raw, grouped_data, sig_res_map, config, sub_names):
    plt.rcParams['font.family'] = 'sans-serif'
    
    n_major = len(grouped_data)
    n_sub = len(sub_names)
    
    # 幅計算
    fig_w = config['width'] if config['width'] > 0 else max(6.0, n_major * n_sub * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, config['height']))
    
    x_base = np.arange(n_major)
    w = config['bar_width']
    # 棒の中心オフセット計算
    total_group_width = w * n_sub * 1.1 # 1.1は棒間の隙間
    offsets = np.linspace(-total_group_width/2 + w/2, total_group_width/2 - w/2, n_sub)
    
    all_raw = df_raw['Val'].tolist()
    max_v = max(all_raw) if all_raw else 1
    # Log safety
    pos_vals = [x for x in all_raw if x > 0]
    min_pos_v = min(pos_vals) if pos_vals else 0.01
    
    name_to_x_map = {}
    base_y_map = {} 
    is_log = config['scale'] == "対数 (Log)"
    if is_log: ax.set_yscale('log')
    
    # --- Draw Data ---
    for i, s_name in enumerate(sub_names):
        col = config['colors'].get(s_name, "#333333")
        
        # Gather data
        means, errs, raw_vals_list = [], [], []
        x_coords = x_base + offsets[i]
        
        for j, m_group in enumerate(grouped_data.keys()):
            v = grouped_data[m_group].get(s_name, [])
            if is_log: v, _ = clean_data_for_log(v)
            else: v = v if isinstance(v, list) else []
            
            # Map coords
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
            
            # Base Y
            top = max(v) if len(v)>0 else 0
            if "棒" in config['manual_type']: 
                top = (means[-1] + errs[-1]) if len(v)>0 else 0
            margin = 1.2 if is_log else 1.05
            base_y_map[(m_group, s_name)] = top * margin

        # Bar
        if "棒" in config['manual_type']: 
            ax.bar(x_coords, means, width=w, label=s_name, color=col, edgecolor='black', alpha=0.8, yerr=errs, capsize=4, zorder=1)
        else:
            # Boxplot
            for k, v in enumerate(raw_vals_list):
                if len(v) > 0:
                    ax.boxplot(v, positions=[x_coords[k]], widths=w*0.8, patch_artist=True, 
                               boxprops=dict(facecolor=col, alpha=0.8), medianprops=dict(color='black'), showfliers=False)

        # Scatter
        for k, v in enumerate(raw_vals_list):
            if len(v) > 0:
                noise = np.random.normal(0, config['jitter']*0.05, len(v)) # 2要因は狭いのでJitter控えめ
                ax.scatter(x_coords[k] + noise, v, s=config['dot_size'], facecolors='white', edgecolors='#555555', zorder=3, alpha=config['dot_alpha'])

    # --- Sig Bars (Cluster Local Tetris) ---
    global_max_y = max_v
    step_y = max_v * 0.1
    
    for m_group in grouped_data.keys():
        pairs = sig_res_map.get(m_group, [])
        if not pairs: continue
        
        # Local Map
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
    
    # Legend
    if "棒" in config['manual_type']:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=config['colors'].get(n,'#333'), edgecolor='black', label=n) for n in sub_names]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if config['manual_y_max'] > 0:
        ax.set_ylim(bottom=None, top=config['manual_y_max'])
    else:
        top_margin = 1.1 if not is_log else 1.5
        bottom_val = 0 if not is_log else min_pos_v * 0.5
        ax.set_ylim(bottom=bottom_val, top=global_max_y * top_margin)

    return fig

# ---------------------------------------------------------
# 2. サイドバー設定
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 【重要：論文・学会発表での使用】")
    st.warning("""
    **研究成果として公表される予定ですか？**
    本ツールは現在ベータ版です。学術利用の際は**必ず事前に開発者（金子）までご連絡ください。**
    共著（Co-authorship）や謝辞（Acknowledgment）についてご相談させていただきます。
    👉 **[連絡・お問い合わせ](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)
    st.divider()

    analysis_mode = st.radio("解析モード", ["1要因 (単純比較)", "2要因 (二元配置分散分析)"], 
                             help="1要因: A vs B vs C\n2要因: 要因A × 要因B")
    st.divider()

    st.header("🛠️ グラフ設定")
    with st.expander("📈 種類・スケール", expanded=True):
        if analysis_mode.startswith("1要因"):
            graph_mode_ui = st.radio("選択モード", ["自動 (Auto - 推奨)", "手動 (Manual)"])
            scale_option = st.radio("Y軸スケール", ["線形 (Linear)", "対数 (Log)"])
            auto_zoom = st.checkbox("外れ値除外ズーム", value=False) if scale_option.startswith("線形") else False
            
            manual_graph_type = "棒グラフ (Bar)"
            error_type = "SD (標準偏差)"
            if graph_mode_ui.startswith("手動"):
                manual_graph_type = st.selectbox("形式", ["棒グラフ (Bar)", "箱ひげ図 (Box)", "バイオリン図 (Violin)"])
                if "棒" in manual_graph_type:
                    error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"])
                else: error_type = "None"
            else:
                st.caption("※ 分布に基づき自動選択")
                error_type = "SD (標準偏差)"
        else:
            graph_type_2way = st.selectbox("形式", ["棒グラフ (Bar)", "箱ひげ図 (Box)"])
            error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"]) if "棒" in graph_type_2way else "None"
            scale_option = st.radio("Y軸スケール", ["線形 (Linear)", "対数 (Log)"])
            graph_mode_ui = "手動"; manual_graph_type = graph_type_2way; auto_zoom = False

    with st.expander("🎨 デザイン微調整", expanded=False):
        fig_title = st.text_input("タイトル", value="Experiment Result")
        y_axis_label = st.text_input("Y軸ラベル", value="Relative Value")
        manual_y_max = st.number_input("Y軸最大 (0で自動)", value=0.0, step=1.0)
        st.divider()
        manual_width = st.slider("画像の幅 (0で自動)", 0.0, 20.0, 0.0, 0.5)
        fig_height = st.slider("画像の高さ", 3.0, 15.0, 6.0)
        bar_width = st.slider("棒の太さ", 0.1, 1.0, 0.35, 0.05)
        # 間隔調整: 2要因ではクラスター間の距離として機能させる
        group_spacing = st.slider("間隔", 0.5, 3.0, 1.0, 0.1) if analysis_mode.startswith("1要因") else 1.0
        
        st.caption("ドット・その他")
        dot_size = st.slider("ドットサイズ", 0, 100, 20)
        dot_alpha = st.slider("ドット透明度", 0.1, 1.0, 0.7)
        jitter = st.slider("Jitter (散らし)", 0.0, 1.0, 0.2)

# ---------------------------------------------------------
# 3. メインエリア：データ入力
# ---------------------------------------------------------
st.title("🔬 Ultimate Sci-Stat & Graph Engine V13 (Matplotlib)")

plot_config = {
    'mode': graph_mode_ui, 'manual_type': manual_graph_type, 'scale': scale_option,
    'error': error_type, 'auto_zoom': auto_zoom, 'title': fig_title, 'ylabel': y_axis_label,
    'width': manual_width, 'height': fig_height, 'bar_width': bar_width, 'spacing': group_spacing,
    'dot_size': dot_size, 'dot_alpha': dot_alpha, 'jitter': jitter, 'colors': {}, 'manual_y_max': manual_y_max
}

data_dict = {}
grouped_data = {}

# === 1要因入力 ===
if analysis_mode.startswith("1要因"):
    st.caption("1つの条件で複数の群を比較します")
    t1, t2 = st.tabs(["✍️ 手動入力", "📂 CSVアップロード"])
    with t1:
        if 'g_cnt' not in st.session_state: st.session_state.g_cnt = 3
        c1, c2 = st.columns([1,5])
        if c1.button("＋"): st.session_state.g_cnt += 1
        if c2.button("－"): st.session_state.g_cnt = max(2, st.session_state.g_cnt - 1)
        cols = st.columns(min(st.session_state.g_cnt, 4))
        for i in range(st.session_state.g_cnt):
            with cols[i%4]:
                name = st.text_input(f"Group {i+1}", f"Group {i+1}", key=f"n{i}")
                raw = st.text_area(f"値 {i+1}", key=f"d{i}")
                v = parse_vals(raw); 
                if v: data_dict[name] = v
    with t2:
        up = st.file_uploader("CSVファイル", type="csv")
        if up:
            try:
                df = pd.read_csv(up)
                st.write("プレビュー:", df.head(3))
                if st.radio("形式", ["縦持ち", "横持ち (一括)"]).startswith("縦"):
                    cols = df.columns.tolist()
                    c_grp = st.selectbox("G列", cols); c_val = st.selectbox("V列", [c for c in cols if c!=c_grp])
                    if st.button("読込"):
                        for g in df[c_grp].unique():
                            v = df[df[c_grp]==g][c_val].dropna().tolist()
                            clean = [float(x) for x in v if str(x).replace('.','').isdigit()]
                            if clean: data_dict[g] = clean
                else:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    sel = st.multiselect("列選択", num_cols, default=list(num_cols)[:3])
                    if st.button("読込"):
                        for c in sel:
                            v = df[c].dropna().tolist(); 
                            if v: data_dict[c] = v
            except Exception as e: st.error(str(e))

# === 2要因入力 ===
else:
    st.caption("2要因 (Factor A × Factor B) の交互作用解析")
    c1, c2 = st.columns(2)
    with c1:
        mj_str = st.text_area("要因A (X軸) ※改行区切り", "DMSO\nDrug_X\nDrug_Y", height=100)
        mj_grps = [x.strip() for x in mj_str.split('\n') if x.strip()]
    with c2:
        if 'sub_cnt' not in st.session_state: st.session_state.sub_cnt = 2
        sc1, sc2 = st.columns(2)
        if sc1.button("＋サブ群"): st.session_state.sub_cnt += 1
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
# 4. カラー設定
# ---------------------------------------------------------
with st.sidebar:
    with st.expander("🖍️ カラー設定", expanded=True):
        defs = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
        if analysis_mode.startswith("1要因") and data_dict:
            for i, k in enumerate(data_dict.keys()):
                plot_config['colors'][k] = st.color_picker(k, defs[i%len(defs)])
        elif analysis_mode.startswith("2要因") and 'sub_names' in locals():
            for i, k in enumerate(sub_names):
                plot_config['colors'][k] = st.color_picker(k, defs[i%len(defs)])

# ---------------------------------------------------------
# 5. 実行 (Report & Draw)
# ---------------------------------------------------------
if analysis_mode.startswith("1要因"):
    if len(data_dict) >= 2 and check_data_validity(data_dict.values()):
        # Calc
        p_val, method, is_norm, reason = auto_select_test(list(data_dict.values()))
        st.success(f"解析完了: {method}")
        with st.expander("📝 詳細レポート", expanded=True):
            st.markdown(f"**選定根拠**: {reason} -> **{method}**")
            st.markdown(f"**P値**: {p_val:.4e} ({'有意差あり' if p_val < 0.05 else '有意差なし'})")
            st.code(f"Statistical analyses were performed using Python ({method}).")

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
                st.warning("scikit-posthocs未導入。代替ロジック(Bonferroni-MannWhitney)を実行")
                sig_pairs = run_fallback_posthoc(vals, grps)
        
        # Draw (Matplotlib)
        try:
            fig = draw_matplotlib_1factor(data_dict, sig_pairs, plot_config, is_norm)
            st.pyplot(fig)
            buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            st.download_button("📥 画像を保存 (PNG)", buf, file_name="result.png", mime="image/png")
        except Exception as e: st.error(f"描画エラー: {e}")
    else: st.info("データを入力してください")

else: # 2要因
    if len(grouped_data) > 0:
        rows = []
        for m, sub in grouped_data.items():
            for s, v in sub.items():
                for x in v: rows.append({'A': m, 'B': s, 'Val': x})
        df_a = pd.DataFrame(rows)
        
        if not df_a.empty:
            st.header("解析結果")
            # ANOVA
            try:
                model = ols('Val ~ C(A) * C(B)', data=df_a).fit()
                res = sm.stats.anova_lm(model, typ=2)
                p_int = res.loc['C(A):C(B)', 'PR(>F)']
                with st.expander("📊 ANOVA結果", expanded=False):
                    st.write(res)
                    st.info(f"交互作用: **{'あり' if p_int < 0.05 else 'なし'}** (P={p_int:.4f})")
                    fig_i, ax_i = plt.subplots()
                    interaction_plot(x=df_a['A'], trace=df_a['B'], response=df_a['Val'], ax=ax_i)
                    st.pyplot(fig_i)
            except: st.warning("ANOVA計算不可")

            # Simple Effects
            sig_res_map = {}
            st.subheader("単純主効果 (層別解析)")
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
            except Exception as e: st.error(f"描画エラー: {e}")
    else: st.info("データを入力してください")

# ---------------------------------------------------------
# 6. 免責事項
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    st.caption("【免責事項】")
    st.caption("""
    本ソフトウェアは研究用ツールとして「現状有姿」で提供されます。
    開発者は、本ツールの計算結果の正確性、完全性、特定目的への適合性について一切の保証を行いません。
    本ツールの使用により生じた、いかなる損害についても、開発者は責任を負いません。
    """)
