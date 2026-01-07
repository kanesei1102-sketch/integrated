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
st.set_page_config(
    page_title="Ultimate Sci-Stat V15 (Report-Enhanced)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1. 共通関数 (Logic)
# ---------------------------------------------------------

def parse_vals(text):
    """テキストエリアからの数値をパース（全角数字対応）"""
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
    """対数スケール用に0以下の値を除外"""
    arr = np.array(vals)
    positive = arr[arr > 0]
    if len(positive) < len(arr):
        return positive.tolist(), True
    return positive.tolist(), False

def check_data_validity(values_list):
    """各群に最低2つのデータがあるかチェック"""
    if not values_list: return False
    return all(len(v) >= 2 for v in values_list)

def get_sig_label(p):
    """P値をアスタリスクに変換"""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def run_fallback_posthoc(groups_vals, group_names):
    """scikit-posthocsがない場合の代替処理（Mann-Whitney + Bonferroni）"""
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
    """
    統計検定の自動選択ロジック
    Returns: p_val, method_name, is_parametric, context_dict
    """
    context = {
        "small_n": False,
        "all_normal": True,
        "is_equal_var": True,
        "posthoc": "なし"
    }

    if not check_data_validity(groups_vals):
        return 1.0, "データ不足", False, context

    # 1. N数チェック (n<3は正規性検定が不安定なためフラグ立て)
    if any(len(v) < 3 for v in groups_vals):
        context["small_n"] = True

    # 2. 正規性検定 (Shapiro-Wilk)
    for v in groups_vals:
        if len(v) >= 3:
            if stats.shapiro(v)[1] <= 0.05: context["all_normal"] = False
    
    # 3. 等分散性検定 (Levene)
    try: _, p_lev = stats.levene(*groups_vals); context["is_equal_var"] = (p_lev > 0.05)
    except: context["is_equal_var"] = True

    method_name = ""
    p_val = 1.0

    if len(groups_vals) == 2:
        context["posthoc"] = "-"
        if context["all_normal"]:
            if context["is_equal_var"]:
                method_name = "Student's t-test"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=True)
            else:
                method_name = "Welch's t-test"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=False)
        else:
            method_name = "Mann-Whitney U test"
            _, p_val = stats.mannwhitneyu(groups_vals[0], groups_vals[1], alternative='two-sided')
    else:
        if context["all_normal"] and context["is_equal_var"]:
            method_name = "One-way ANOVA"
            context["posthoc"] = "Tukey-Kramer test"
            _, p_val = stats.f_oneway(*groups_vals)
        else:
            method_name = "Kruskal-Wallis test"
            context["posthoc"] = "Dunn's test (Bonferroni)" if HAS_POSTHOCS else "Mann-Whitney U (Bonferroni)"
            _, p_val = stats.kruskal(*groups_vals)

    return p_val, method_name, context["all_normal"], context

def calculate_sig_bars_layout(pairs, name_to_x, base_y_map, step_y, is_log):
    """有意差バーの高さ計算（重なり回避ロジック）"""
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
# 2. 描画関数 (Matplotlib)
# ---------------------------------------------------------

def draw_matplotlib_1factor(data_dict, sig_pairs, config, is_norm):
    # グラフ内は英語フォント推奨
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    group_names = list(data_dict.keys())
    all_values = list(data_dict.values())
    
    # Spacing調整
    x_pos = np.arange(len(group_names)) * config['spacing']
    name_to_x = {name: x for name, x in zip(group_names, x_pos)}

    # 図の幅調整
    base_width_per_group = 3.0
    fig_w = max(6.0, len(data_dict) * base_width_per_group * config['spacing'])
    
    fig, ax = plt.subplots(figsize=(fig_w, config['height']))
    
    all_flat = [x for sub in all_values for x in sub]
    max_v = max(all_flat) if all_flat else 1
    pos_vals = [x for x in all_flat if x > 0]
    min_pos_v = min(pos_vals) if pos_vals else 0.01
    
    base_y_map = {} 
    # 自動選択時のグラフタイプ
    final_type = config['manual_type'] if config['mode'].startswith("手動") else ("箱ひげ図 (Box)" if not is_norm else "棒グラフ (Bar)")

    for i, (name, vals) in enumerate(data_dict.items()):
        vals = np.array(vals); p = x_pos[i]
        
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
        
        top_val = max(vals_plot) if len(vals_plot)>0 else 0
        if "棒" in final_type:
            top_val = mean + (err if config['error'] != "None" else 0)
        
        margin_ratio = 1.05 if config['scale'].startswith("線形") else 1.2
        base_y_map[name] = top_val * margin_ratio
        
        # Plotting
        if "棒" in final_type:
            ax.bar(p, mean, width=config['bar_width'], color=col, edgecolor='black', alpha=0.8, zorder=1)
            if config['error'] != "None":
                ax.errorbar(p, mean, yerr=err, fmt='none', c='black', capsize=5, zorder=2)
        elif "箱" in final_type and len(vals_plot)>0:
            ax.boxplot(vals_plot, positions=[p], widths=config['bar_width'], patch_artist=True, 
                       boxprops=dict(facecolor=col, alpha=0.8), medianprops=dict(color='black'), showfliers=False, zorder=1)
        elif "バイオリン" in final_type and len(vals_plot)>0:
            parts = ax.violinplot(vals_plot, positions=[p], widths=config['bar_width'], showextrema=False)
            for pc in parts['bodies']: pc.set_facecolor(col); pc.set_alpha(0.8); pc.set_zorder(1)
            
        # Jitter (散らし)
        if len(vals_plot) > 0:
            jitter_width = config['bar_width'] * 0.4 * config['jitter'] 
            noise = np.random.uniform(-1, 1, len(vals_plot)) * jitter_width
            ax.scatter(p + noise, vals_plot, s=config['dot_size'], facecolors='white', edgecolors='#555555', zorder=3, alpha=config['dot_alpha'])

    # Significance Bars
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
    
    min_x = min(x_pos) - 1.0 
    max_x = max(x_pos) + 1.0 
    ax.set_xlim(min_x, max_x)
    
    if config['manual_y_max'] > 0:
        ax.set_ylim(bottom=None, top=config['manual_y_max'])
    else:
        top_margin = 1.1 if not is_log else 1.5
        bottom_val = 0 if not is_log else min_pos_v * 0.5
        if config['scale'].startswith("線形") and config['auto_zoom']:
             ax.set_ylim(bottom=0, top=global_max_y * top_margin)
        else:
             ax.set_ylim(bottom=bottom_val, top=global_max_y * top_margin)
    
    plt.tight_layout()
    return fig

def draw_matplotlib_2factor(df_raw, grouped_data, sig_res_map, config, sub_names):
    # 2要因グラフ設定
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

    n_major = len(grouped_data)
    n_sub = len(sub_names)
    
    x_base = np.arange(n_major) * config['spacing']
    
    base_width_per_major = max(4.0, n_sub * 1.5)
    fig_w = max(6.0, n_major * base_width_per_major * config['spacing'])
    
    fig, ax = plt.subplots(figsize=(fig_w, config['height']))
    
    w = config['bar_width']
    total_group_width = w * n_sub * 1.2 
    offsets = np.linspace(-total_group_width/2 + w/2 + (w*0.1), total_group_width/2 - w/2 - (w*0.1), n_sub)
    
    all_raw = df_raw['Val'].tolist()
    max_v = max(all_raw) if all_raw else 1
    pos_vals = [x for x in all_raw if x > 0]
    min_pos_v = min(pos_vals) if pos_vals else 0.01
    
    name_to_x_map = {}
    base_y_map = {} 
    is_log = config['scale'] == "対数 (Log)"
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
            if "棒" in config['manual_type']: top = (means[-1] + errs[-1]) if len(v)>0 else 0
            margin = 1.2 if is_log else 1.05
            base_y_map[(m_group, s_name)] = top * margin

        if "棒" in config['manual_type']: 
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
    
    if "棒" in config['manual_type']:
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

    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 2. サイドバー設定 (日本語・権利主張版)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 【重要：論文・学会発表での使用】")
    st.warning("""
    **研究成果として公表される予定ですか？**
    本ツールは学術研究を支援するために開発されました。
    論文投稿の際は、Methodsセクションへの引用や謝辞をご検討ください。
    高解像度のベクター画像が必要な場合はご連絡ください。
    👉 **[連絡・お問い合わせ](https://forms.gle/ExampleFormID)**
    """)
    st.divider()

    analysis_mode = st.radio("解析モード", ["1要因 (単純比較)", "2要因 (二元配置分散分析)"], 
                             help="1要因: A vs B vs C\n2要因: 要因A × 要因B")
    st.divider()

    st.header("🛠️ グラフ設定 (Graph Config)")
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

    with st.expander("🎨 デザイン・ラベリング (Edit)", expanded=True):
        st.caption("※ ここで英語ラベルを設定すると、そのままグラフに反映されます。")
        fig_title = st.text_input("グラフタイトル (Title)", value="Experiment Result")
        y_axis_label = st.text_input("Y軸ラベル (Y-Axis)", value="Relative Value", help="例: Nuclei Count, Cell Area などに変更可能")
        manual_y_max = st.number_input("Y軸最大値 (0で自動)", value=0.0, step=1.0)
        
        st.divider()
        st.caption("スタイル調整")
        fig_height = st.slider("画像の高さ", 3.0, 15.0, 6.0)
        bar_width = st.slider("棒の太さ", 0.1, 1.0, 0.35, 0.05)
        
        group_spacing = st.slider(
            "間隔の広さ (1.0=最大)", 
            0.2, 1.0, 1.0, 0.05, 
            help="1.0で最大間隔。値を小さくすると棒の太さを保ったまま間隔を詰めます。"
        ) if analysis_mode.startswith("1要因") else st.slider(
            "群間の間隔 (1.0=最大)", 
            0.2, 1.0, 1.0, 0.05
        )
        
        st.caption("ドット (Jitter)")
        dot_size = st.slider("サイズ", 0, 20, 6)
        dot_alpha = st.slider("透明度 (Alpha)", 0.1, 1.0, 0.7)
        jitter = st.slider("散らし具合 (Jitter)", 0.0, 1.0, 0.2)

# ---------------------------------------------------------
# 3. メインエリア：データ入力
# ---------------------------------------------------------
st.title("🔬 Ultimate Sci-Stat & Graph Engine V15")
st.caption("Data Integrity / Traceability / Publication-Ready Visualization")

plot_config = {
    'mode': graph_mode_ui, 'manual_type': manual_graph_type, 'scale': scale_option,
    'error': error_type, 'auto_zoom': auto_zoom, 'title': fig_title, 'ylabel': y_axis_label,
    'width': 0, 'height': fig_height, 'bar_width': bar_width, 'spacing': group_spacing,
    'dot_size': dot_size, 'dot_alpha': dot_alpha, 'jitter': jitter, 'colors': {}, 'manual_y_max': manual_y_max
}

data_dict = {}
grouped_data = {}

# === 1要因入力 ===
if analysis_mode.startswith("1要因"):
    st.info("💡 **Hint:** CSVをアップロードするか、値を直接貼り付けてください。QuantData形式も自動認識します。")
    t1, t2 = st.tabs(["✍️ 手動入力", "📂 CSVアップロード"])
    
    if 'csv_data_cache_jp' not in st.session_state:
        st.session_state.csv_data_cache_jp = {}

    with t1:
        if 'g_cnt' not in st.session_state: st.session_state.g_cnt = 3
        c1, c2 = st.columns([1,5])
        if c1.button("＋ 群を追加"): st.session_state.g_cnt += 1
        if c2.button("－ 群を削除"): st.session_state.g_cnt = max(2, st.session_state.g_cnt - 1)
        
        cols = st.columns(min(st.session_state.g_cnt, 4))
        for i in range(st.session_state.g_cnt):
            with cols[i%4]:
                name = st.text_input(f"Group {i+1} Name", f"Group_{i+1}", key=f"n{i}", help="グラフ内ではそのまま表示されます (英語推奨)")
                raw = st.text_area(f"Values {i+1}", key=f"d{i}", height=150)
                v = parse_vals(raw); 
                if v: data_dict[name] = v
    with t2:
        up = st.file_uploader("CSVファイル (QuantData/解析ツール出力対応)", type="csv")
        if up:
            try:
                df = pd.read_csv(up)
                
                # --- QuantData Specific Logic (Auto-Load with Selectors) ---
                if 'Main_Value' in df.columns and 'Group' in df.columns:
                    st.success("✅ QuantData形式を検出しました (Detection: Specific Format)")
                    
                    # Column Selectors
                    c1, c2 = st.columns(2)
                    
                    # X-axis (Grouping)
                    cols = df.columns.tolist()
                    def_grp = 'Group' if 'Group' in cols else cols[0]
                    c_grp = c1.selectbox("群分け (X軸)", cols, index=cols.index(def_grp))
                    
                    # Y-axis (Value)
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'Main_Value' in num_cols: def_val = 'Main_Value'
                    else: def_val = num_cols[0] if num_cols else None
                    
                    if num_cols:
                        c_val = c2.selectbox("測定値 (Y軸)", num_cols, index=num_cols.index(def_val) if def_val in num_cols else 0)
                    else:
                        st.error("数値列が見つかりません")
                        c_val = None

                    # Auto-Load Logic (No button needed)
                    if c_val:
                        temp = {}
                        try:
                            unique_groups = df[c_grp].unique()
                        except:
                            unique_groups = df[c_grp].astype(str).unique()

                        for g in unique_groups:
                            sub_df = df[df[c_grp] == g]
                            raw_v = sub_df[c_val].dropna().tolist()
                            clean = [float(x) for x in raw_v if str(x).replace('.','').replace('-','').isdigit()]
                            if clean: temp[str(g)] = clean
                        
                        st.session_state.csv_data_cache_jp = temp
                        data_dict.update(temp)
                        st.success(f"データ読込完了: {len(temp)}群 (X: {c_grp}, Y: {c_val})")

                # --- Standard Format Logic (Group, Value) ---
                elif "Group" in df.columns and "Value" in df.columns:
                    st.success("✅ 標準フォーマットを検出しました (Group, Value)")
                    temp = {}
                    for g in df["Group"].unique():
                        v = df[df["Group"]==g]["Value"].dropna().tolist()
                        clean = [float(x) for x in v if str(x).replace('.','').isdigit()]
                        if clean: temp[g] = clean
                    
                    st.session_state.csv_data_cache_jp = temp
                    data_dict.update(temp)
                    st.success(f"データ読込完了: {len(temp)}群")

                # --- Generic CSV Logic (Select Columns) ---
                else:
                    st.divider()
                    st.warning("⚠️ 専用フォーマットではありません。列を指定して読み込みます。")
                    cols = df.columns.tolist()
                    
                    c1, c2 = st.columns(2)
                    c_grp = c1.selectbox("群分けの列 (Group)", cols, index=0)
                    c_val = c2.selectbox("数値の列 (Value)", [c for c in cols if c!=c_grp], index=len(cols)-1 if len(cols)>1 else 0)
                    
                    # Manual load needed here as user might be deciding
                    # Using session state to mimic auto-load after selection
                    if st.button("指定した列で読み込む"):
                        temp = {}
                        for g in df[c_grp].unique():
                            v = df[df[c_grp]==g][c_val].dropna().tolist()
                            clean = [float(x) for x in v if str(x).replace('.','').isdigit()]
                            if clean: temp[str(g)] = clean
                        
                        st.session_state.csv_data_cache_jp = temp
                        st.rerun() 

                # Always apply cache if exists
                if st.session_state.csv_data_cache_jp and not data_dict:
                    data_dict.update(st.session_state.csv_data_cache_jp)
                
                if st.session_state.csv_data_cache_jp:
                    if st.button("データをクリアしてリセット"):
                        st.session_state.csv_data_cache_jp = {}
                        st.rerun()

            except Exception as e: st.error(f"エラー: {str(e)}")

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
            sub_names.append(st.text_input(f"Sub {i+1} (凡例)", f"Time_{i+1}", key=f"s{i}"))
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
    with st.expander("🖍️ カラー設定 (Colors)", expanded=False):
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
        # Calc & Context
        p_val, method, is_norm, ctx = auto_select_test(list(data_dict.values()))
        st.markdown("---")
        st.subheader(f"📊 解析結果 (Method: {method})")
        
        # --- Report Logic (Enhanced) ---
        easy_reason = ""
        if ctx["all_normal"] and ctx["is_equal_var"]:
            easy_reason = "データ分布に有意な偏りは認められず、かつ分散の等質性も棄却されなかったため、最も統計的検出力の高い標準的な『パラメトリック検定』を選択しました。"
        elif not ctx["all_normal"]:
            easy_reason = "データ分布に非正規性または外れ値の可能性が示唆されたため、順位に基づく『ノンパラメトリック検定』を選択しました。"
        else:
            easy_reason = "正規性は棄却されませんでしたが、分散の等質性が棄却されたため、分散不均一に対して頑健な手法を選択しました。"

        if ctx["small_n"]:
            easy_reason += "\n    ※ 一部の群でサンプルサイズが小さいため、分布評価の精度には制限があります。"

        result_summary = "【有意差あり】" if p_val < 0.05 else "【有意差なし】"
        conclusion_text = (
            "本データセットにおいて群間に統計学的に有意な差が認められ、少なくとも一部の群間で平均値（または中央値）の差が存在することが示唆されました。"
            if p_val < 0.05
            else
            "本データセットにおいて群間に統計学的に有意な差は認められず、明確な平均値の差は確認できませんでした。"
        )

        norm_res_text = "有意な偏りなし（棄却されず）" if ctx["all_normal"] else "非正規性が示唆（棄却）"
        if ctx["small_n"]:
            norm_res_text += " ※参考値（n<3）"
        var_res_text = "分散の等質性：棄却されず" if ctx["is_equal_var"] else "分散の等質性：棄却"

        analysis_path = f"""
【統計手法選択プロセス（自動判定）】
1. 正規性検定（Shapiro-Wilk検定）：{norm_res_text}
2. 分散の等質性検定（Levene検定）：{var_res_text}
⇒ 上記の診断結果に基づき、**{method}** を採用しました。
"""

        with st.expander("📝 そのまま使える解析レポート（詳細）", expanded=True):
            full_report = f"""
【解析レポート：{", ".join(data_dict.keys())} の比較】{analysis_path}

1. 検定法選択の根拠：
   使用手法：{method}
   理由：{easy_reason}

2. 解析結果：
   判定：{result_summary}
   全体P値：{p_val:.4e}
   （有意水準 α = 0.05）

3. 事後解析結果：
   {"多重比較検定を実施し、その有意性をグラフ上に反映しました。" if len(data_dict) > 2 else "2群間の直接比較を行いました。"}

4. 結論：
   {conclusion_text}
            """
            st.text_area("レポート全文", value=full_report, height=400)

        with st.expander("📄 論文用『Methods』セクション草案", expanded=False):
            methods_text = f"""
統計解析は Python（SciPy ライブラリ）を用いて実施しました。
データの正規性は Shapiro-Wilk 検定により評価し、分散の等質性は Levene 検定により評価しました。
群間比較には {method} を用いました。
{f"（事後解析：{ctx['posthoc']}）" if len(data_dict) > 2 else ""}
P値が 0.05 未満の場合を統計学的に有意と判断しました。
            """
            st.text_area("Methods 草案", value=methods_text, height=150)
        
        # --- End of Report Logic ---

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
                sig_pairs = run_fallback_posthoc(vals, grps)
        
        # Draw (Matplotlib)
        try:
            fig = draw_matplotlib_1factor(data_dict, sig_pairs, plot_config, is_norm)
            st.pyplot(fig)
            
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                st.download_button("📥 高画質PNGを保存", buf, file_name="result_high_res.png", mime="image/png")
            with c_dl2:
                buf_svg = io.BytesIO(); fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                st.download_button("📥 ベクター画像 (SVG) を保存", buf_svg, file_name="result_vector.svg", mime="image/svg+xml")
                
        except Exception as e: st.error(f"描画エラー: {e}")
    else: st.info("👈 左側のサイドバー、または上のタブからデータを入力してください")

else: # 2要因
    if len(grouped_data) > 0:
        rows = []
        for m, sub in grouped_data.items():
            for s, v in sub.items():
                for x in v: rows.append({'A': m, 'B': s, 'Val': x})
        df_a = pd.DataFrame(rows)
        
        if not df_a.empty:
            st.header("解析結果 (Two-way ANOVA)")
            try:
                model = ols('Val ~ C(A) * C(B)', data=df_a).fit()
                res = sm.stats.anova_lm(model, typ=2)
                p_int = res.loc['C(A):C(B)', 'PR(>F)']
                with st.expander("📊 ANOVA詳細テーブル", expanded=False):
                    st.write(res)
                    st.info(f"交互作用 (Interaction): **{'あり' if p_int < 0.05 else 'なし'}** (P={p_int:.4f})")
            except: st.warning("ANOVA計算不可")

            sig_res_map = {}
            for m, sub in grouped_data.items():
                s_keys = list(sub.keys()); s_vals = list(sub.values())
                if not check_data_validity(s_vals): continue
                p, method, _, _ = auto_select_test(s_vals)
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
                        sig_res_map[m] = run_fallback_posthoc(s_vals, s_keys)
            
            # Draw (Matplotlib)
            try:
                fig = draw_matplotlib_2factor(df_a, grouped_data, sig_res_map, plot_config, sub_names)
                st.pyplot(fig)
                
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                    st.download_button("📥 高画質PNGを保存", buf, file_name="result_2way_high.png", mime="image/png")
                with c_dl2:
                    buf_svg = io.BytesIO(); fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                    st.download_button("📥 ベクター画像 (SVG) を保存", buf_svg, file_name="result_2way.svg", mime="image/svg+xml")

            except Exception as e: st.error(f"描画エラー: {e}")
    else: st.info("データを入力してください")

# ---------------------------------------------------------
# 6. サイドバー最下部：免責事項 (JP Final)
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    st.caption("【免責事項 / Disclaimer】")
    st.caption("""
    本ソフトウェアは研究用ツールとして「現状有姿」で提供されます。
    統計学的妥当性の最終判断は、必ず利用者の責任において行ってください。
    本ツールの使用により生じたいかなる損害についても、開発者は責任を負いません。
    """)
