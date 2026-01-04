import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
st.set_page_config(page_title="Ultimate Sci-Stat V11 (Interactive)", layout="wide")

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
        return positive.tolist(), True # 除外あり
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
    """Fallback Posthoc (Bonferroni-MannWhitney)"""
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
    """統計検定の自動選択ロジック"""
    if not check_data_validity(groups_vals):
        return 1.0, "データ不足 (N<2)", False

    all_normal = True
    for v in groups_vals:
        if len(v) >= 3:
            if stats.shapiro(v)[1] <= 0.05: all_normal = False
    
    try: _, p_lev = stats.levene(*groups_vals); is_equal_var = (p_lev > 0.05)
    except: is_equal_var = True

    method_name = ""
    p_val = 1.0

    if len(groups_vals) == 2:
        if all_normal:
            if is_equal_var:
                method_name = "Student's t-test"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=True)
            else:
                method_name = "Welch's t-test"
                _, p_val = stats.ttest_ind(groups_vals[0], groups_vals[1], equal_var=False)
        else:
            method_name = "Mann-Whitney U"
            _, p_val = stats.mannwhitneyu(groups_vals[0], groups_vals[1], alternative='two-sided')
    else:
        if all_normal and is_equal_var:
            method_name = "One-way ANOVA"
            _, p_val = stats.f_oneway(*groups_vals)
        else:
            method_name = "Kruskal-Wallis"
            _, p_val = stats.kruskal(*groups_vals)

    return p_val, method_name, all_normal

def calculate_sig_bars_layout(pairs, name_to_x, base_y_map, step_y, is_log):
    """有意差バー配置 (Tetris Algorithm)"""
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
                if not (x2 < occ_x1 - 0.05 or x1 > occ_x2 + 0.05): 
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
# 2. 描画関数 (Plotly Interactive)
# ---------------------------------------------------------

def draw_plotly_1factor(data_dict, sig_pairs, config, is_norm):
    group_names = list(data_dict.keys())
    
    fig = go.Figure()
    
    # データ処理 & 描画
    all_flat = []
    base_y_map = {} # for sig bars
    
    for i, name in enumerate(group_names):
        vals = data_dict[name]
        
        # Log Safety Check
        if config['scale'] == "対数 (Log)":
            vals, removed = clean_data_for_log(vals)
            if removed: st.toast(f"⚠️ {name}: 0以下の値を除外しました (Log Scale)", icon="ℹ️")
            if not vals: continue

        all_flat.extend(vals)
        col = config['colors'].get(name, "#333333")
        
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals)>1 else 0
        sem = std/np.sqrt(len(vals)) if len(vals)>0 else 0
        err = sem if config['error'].startswith("SEM") else std
        
        # Determine Base Y for stacking
        top_val = max(vals) if vals else 0
        if "棒" in config['manual_type'] or (config['mode']=="自動" and is_norm):
             top_val = mean + (err if config['error'] != "None" else 0)
        margin = 1.2 if config['scale']=="対数 (Log)" else 1.05
        base_y_map[name] = top_val * margin

        # Main Trace
        show_legend = False # 1要因はX軸で分かるので凡例不要
        final_type = config['manual_type'] if config['mode'].startswith("手動") else ("箱ひげ" if not is_norm else "棒")

        if "棒" in final_type:
            fig.add_trace(go.Bar(
                x=[name], y=[mean], name=name, marker_color=col,
                error_y=dict(type='data', array=[err], visible=(config['error']!="None")),
                showlegend=show_legend, opacity=0.8
            ))
        elif "箱" in final_type:
            fig.add_trace(go.Box(
                y=vals, name=name, marker_color=col, boxpoints=False, # Points handled separately
                showlegend=show_legend, line=dict(color='black', width=1.5), fillcolor=col
            ))
        elif "バイオリン" in final_type:
            fig.add_trace(go.Violin(
                y=vals, name=name, line_color=col, points=False,
                showlegend=show_legend, meanline_visible=True
            ))

        # Jitter Points (Scatter)
        if vals:
            # PlotlyのBoxにはjitterがあるが、あえてScatterで重ねると制御しやすい
            jitter_x = np.random.normal(0, config['jitter'], size=len(vals))
            # x軸はカテゴリカルなので、内部的には 0, 1, 2... ではなく文字列。
            # Plotlyでjitterさせるには、boxpoints='all' jitter=... を使うのが定石だが、
            # 棒グラフの上に散らす場合は工夫が必要。
            # ここではシンプルに Box/Violin の標準機能を使うか、Barの場合は Scatter を重ねる。
            
            if "棒" in final_type:
                # Barの上に散らすには、Xを数値として扱うか、offsetgroupを使うハックが必要。
                # 簡易的に: 棒グラフでもBox(visible=False, points='all')を重ねて点を出す
                fig.add_trace(go.Box(
                    y=vals, name=name, marker=dict(color='black', size=config['dot_size']/3, opacity=config['dot_alpha']),
                    boxpoints='all', jitter=config['jitter'], pointpos=0, 
                    fillcolor='rgba(0,0,0,0)', line=dict(width=0), showlegend=False, hoverinfo='y'
                ))
            else:
                # Box/Violin は自身のプロパティで点を出す
                fig.update_traces(selector=dict(name=name), boxpoints='all', jitter=config['jitter'], pointpos=0,
                                  marker=dict(color='black', size=config['dot_size']/3, opacity=config['dot_alpha']))

    # Sig Bars
    max_v = max(all_flat) if all_flat else 1
    step_y = max_v * 0.1
    is_log = config['scale'] == "対数 (Log)"
    
    # PlotlyのX軸はカテゴリカル名そのまま。
    # Tetris計算のために index マッピングを作る
    name_to_idx = {name: i for i, name in enumerate(group_names)}
    
    bars = calculate_sig_bars_layout(sig_pairs, name_to_idx, base_y_map, step_y, is_log)
    
    shapes = []
    annotations = []
    
    for b in bars:
        # Plotly shape uses relative coordinates (0 to 1) or data coordinates.
        # Categorical Axis: 0, 1, 2... corresponds to names
        x0, x1, y, label = b['x1'], b['x2'], b['y'], b['label']
        
        # Line shape
        shapes.append(dict(
            type="path",
            path=f"M {x0},{y*0.98} L {x0},{y} L {x1},{y} L {x1},{y*0.98}",
            line=dict(color="black", width=1.5),
            xref="x", yref="y"
        ))
        # Text
        annotations.append(dict(
            x=(x0+x1)/2, y=y, text=label, showarrow=False, yanchor='bottom', font=dict(size=14)
        ))

    # Layout Update
    fig.update_layout(
        title=config['title'],
        yaxis_title=config['ylabel'],
        yaxis_type="log" if is_log else "linear",
        shapes=shapes,
        annotations=annotations,
        width=config['width'] if config['width']>0 else None,
        height=config['height']*100, # Matplotlib inch -> Plotly px conversion approx
        template="simple_white",
        showlegend=False
    )
    
    return fig

def draw_plotly_2factor(df_raw, grouped_data, sig_res_map, config, sub_names):
    # 2要因は Grouped Bar / Box
    fig = go.Figure()
    
    majors = list(grouped_data.keys())
    
    # Base Y Map for sig bars
    # {(Major, Sub): top_y} -> Plotlyのカテゴリ軸は "Major" だが、Subごとのoffsetを知る必要がある
    # Plotlyでは offset を直接取得するのが難しい。
    # 解決策: X軸を "Major" にし、SubGroupを offsetgroup でずらす。
    # Sig Bar のX座標計算が複雑になるため、ここでは「各群の最大値」を取得するに留め、
    # 簡易的に Cluster単位で Sig Bar を描画する（V10同様）
    
    all_flat = []
    
    for s_name in sub_names:
        col = config['colors'].get(s_name, "#333333")
        y_means = []
        y_errs = []
        y_raws = [] # list of lists
        
        for m in majors:
            vals = grouped_data[m].get(s_name, [])
            
            # Log Safety
            if config['scale'] == "対数 (Log)":
                vals, removed = clean_data_for_log(vals)
                if removed: st.toast(f"⚠️ {m}-{s_name}: Log除外あり", icon="ℹ️")

            all_flat.extend(vals)
            y_raws.append(vals)
            
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals)>1 else 0
                sem = std/np.sqrt(len(vals)) if len(vals)>0 else 0
                err = sem if config['error'].startswith("SEM") else std
                y_means.append(mean)
                y_errs.append(err)
            else:
                y_means.append(0); y_errs.append(0)

        # Add Trace
        if "棒" in config['manual_type']:
            fig.add_trace(go.Bar(
                name=s_name, x=majors, y=y_means,
                marker_color=col, opacity=0.8,
                error_y=dict(type='data', array=y_errs, visible=(config['error']!="None")),
                offsetgroup=s_name
            ))
            # Jitter Points on Bar
            # Scatterを重ねる際、xをずらす必要がある。
            # offsetgroupがある場合、x + offset が必要だが Plotly Pythonだけでは計算が面倒。
            # ここではBox(visible=False)トリックを使う
            for idx, m in enumerate(majors):
                vals = y_raws[idx]
                if vals:
                    fig.add_trace(go.Box(
                        name=s_name, x=[m]*len(vals), y=vals,
                        marker=dict(color='black', size=config['dot_size']/3, opacity=config['dot_alpha']),
                        boxpoints='all', jitter=config['jitter'], pointpos=0,
                        fillcolor='rgba(0,0,0,0)', line=dict(width=0), showlegend=False,
                        offsetgroup=s_name, hoverinfo='y'
                    ))

        else:
            # Grouped Boxplot
            # Plotly handles grouped box automatically via x and color/name
            # We need to flatten data for Box trace
            box_x = []
            box_y = []
            for idx, m in enumerate(majors):
                box_x.extend([m]*len(y_raws[idx]))
                box_y.extend(y_raws[idx])
            
            fig.add_trace(go.Box(
                name=s_name, x=box_x, y=box_y,
                marker_color=col,
                boxpoints='all', jitter=config['jitter'], pointpos=0,
                marker=dict(color='black', size=config['dot_size']/3, opacity=config['dot_alpha']),
                line=dict(color='black', width=1.5), fillcolor=col,
                offsetgroup=s_name
            ))

    # --- Sig Bars (Cluster Local) ---
    # PlotlyのGrouped BarにおけるX座標は、各カテゴリ(Major)の中心が整数(0, 1, 2...)
    # offsetgroupの幅や位置は layout.barmode='group' で決まる。
    # デフォルトでは width=0.8 を sub_names数 で割った幅になる。
    
    # 簡易計算:
    n_sub = len(sub_names)
    group_width = 0.8
    bar_w = group_width / n_sub
    # offsets: -0.4 + w/2, ... 
    offsets = np.linspace(-group_width/2 + bar_w/2, group_width/2 - bar_w/2, n_sub)
    
    # Base Y per (Major, Sub)
    base_y_map = {}
    for m in majors:
        for s in sub_names:
            v = grouped_data[m].get(s, [])
            if config['scale']=="対数 (Log)": v, _ = clean_data_for_log(v)
            top = max(v) if v else 0
            if "棒" in config['manual_type']:
                # Err bar logic approx
                if len(v) > 1: top += (np.std(v, ddof=1) if len(v)>1 else 0)
            margin = 1.2 if config['scale']=="対数 (Log)" else 1.05
            base_y_map[(m, s)] = top * margin

    max_v = max(all_flat) if all_flat else 1
    step_y = max_v * 0.1
    is_log = config['scale'] == "対数 (Log)"
    
    shapes = []
    annotations = []
    
    for m_idx, m_group in enumerate(majors):
        pairs = sig_res_map.get(m_group, [])
        if not pairs: continue
        
        # Local Mapping: SubName -> Relative X from m_idx
        # m_idx (0, 1...) is the center of the group
        local_name_to_x = {s: m_idx + offsets[i] for i, s in enumerate(sub_names)}
        local_base_y = {s: base_y_map[(m_group, s)] for s in sub_names}
        
        bars = calculate_sig_bars_layout(pairs, local_name_to_x, local_base_y, step_y, is_log)
        
        for b in bars:
            x0, x1, y, label = b['x1'], b['x2'], b['y'], b['label']
            shapes.append(dict(
                type="path", path=f"M {x0},{y*0.98} L {x0},{y} L {x1},{y} L {x1},{y*0.98}",
                line=dict(color="black", width=1.5), xref="x", yref="y"
            ))
            annotations.append(dict(
                x=(x0+x1)/2, y=y, text=label, showarrow=False, yanchor='bottom', font=dict(size=12)
            ))

    fig.update_layout(
        title=config['title'], yaxis_title=config['ylabel'],
        yaxis_type="log" if is_log else "linear",
        shapes=shapes, annotations=annotations,
        barmode='group',
        width=config['width'] if config['width']>0 else None,
        height=config['height']*100,
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
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
            graph_mode_ui = "手動"; manual_graph_type = graph_type_2way

    with st.expander("🎨 デザイン微調整", expanded=False):
        fig_title = st.text_input("タイトル", value="Experiment Result")
        y_axis_label = st.text_input("Y軸ラベル", value="Relative Value")
        manual_y_max = st.number_input("Y軸最大 (0で自動)", value=0.0, step=1.0)
        st.divider()
        manual_width = st.slider("画像の幅 (0で自動)", 0.0, 2000.0, 0.0, 50.0)
        fig_height = st.slider("画像の高さ", 3.0, 15.0, 6.0)
        bar_width = st.slider("棒の太さ", 0.1, 1.0, 0.35, 0.05)
        group_spacing = st.slider("間隔", 0.5, 3.0, 1.0, 0.1) if analysis_mode.startswith("1要因") else 1.0
        
        st.caption("ドット・その他")
        dot_size = st.slider("ドットサイズ", 0, 20, 6)
        dot_alpha = st.slider("ドット透明度", 0.1, 1.0, 0.7)
        jitter = st.slider("Jitter (散らし)", 0.0, 1.0, 0.3)

# ---------------------------------------------------------
# 3. メインエリア：データ入力
# ---------------------------------------------------------
st.title("🔬 Ultimate Sci-Stat & Graph Engine V11 (Interactive)")

plot_config = {
    'mode': graph_mode_ui, 'manual_type': manual_graph_type, 'scale': scale_option,
    'error': error_type, 'title': fig_title, 'ylabel': y_axis_label,
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
                
                # V11: Smart CSV Loader (Wide Format Support)
                csv_mode = st.radio("データ形式", ["縦持ち (Tidy)", "横持ち (Wide) - 複数列選択"])
                
                if "縦持ち" in csv_mode:
                    cols = df.columns.tolist()
                    c_grp = st.selectbox("グループ列", cols)
                    c_val = st.selectbox("数値列", [c for c in cols if c != c_grp])
                    if st.button("読込"):
                        for g in df[c_grp].unique():
                            v = df[df[c_grp]==g][c_val].dropna().tolist()
                            clean = [float(x) for x in v if str(x).replace('.','').isdigit()]
                            if clean: data_dict[g] = clean
                        st.success("完了")
                else:
                    # Wide: Select Multiple Columns
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    sel_cols = st.multiselect("解析する列を選択", num_cols, default=num_cols[:min(3, len(num_cols))])
                    if st.button("一括読込"):
                        for c in sel_cols:
                            v = df[c].dropna().tolist()
                            if v: data_dict[c] = v
                        st.success(f"{len(data_dict)} グループ読込完了")

            except Exception as e: st.error(f"Error: {e}")

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
                        v = parse_vals(raw)
                        if v: grouped_data[m][s] = v

# ---------------------------------------------------------
# 4. カラー設定
# ---------------------------------------------------------
with st.sidebar:
    with st.expander("🖍️ カラー設定", expanded=True):
        defs = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
        if analysis_mode.startswith("1要因") and data_dict:
            for i, k in enumerate(data_dict.keys()):
                plot_config['colors'][k] = st.color_picker(k, defs[i%len(defs)])
        elif analysis_mode.startswith("2要因") and 'sub_names' in locals():
            for i, k in enumerate(sub_names):
                plot_config['colors'][k] = st.color_picker(k, defs[i%len(defs)])

# ---------------------------------------------------------
# 5. 解析実行 & 描画
# ---------------------------------------------------------
if analysis_mode.startswith("1要因"):
    if len(data_dict) >= 2 and check_data_validity(data_dict.values()):
        # Calc
        p_val, method, is_norm = auto_select_test(list(data_dict.values()))
        st.success(f"解析完了: {method}")
        
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
        
        # Draw Interactive
        try:
            fig = draw_plotly_1factor(data_dict, sig_pairs, plot_config, is_norm)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"描画エラー: {e}")
    else: st.info("データを入力してください (各群 N>=2)")

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
                st.write("▼ 分散分析表"); st.table(res)
                if res.loc['C(A):C(B)', 'PR(>F)'] < 0.05: st.error("⚠️ 交互作用あり")
                else: st.success("✅ 交互作用なし")
            except: st.warning("ANOVA計算不可")

            # Simple Effects
            sig_res_map = {}
            for m, sub in grouped_data.items():
                s_keys = list(sub.keys()); s_vals = list(sub.values())
                if not check_data_validity(s_vals): continue
                
                # 自動選択ロジック
                p, method, _ = auto_select_test(s_vals)
                st.write(f"- **{m}**: P={p:.4f} ({method})")
                
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

            # Draw Interactive
            try:
                fig = draw_plotly_2factor(df_a, grouped_data, sig_res_map, plot_config, sub_names)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"描画エラー: {e}")
    else: st.info("データを入力してください")

# ---------------------------------------------------------
# 6. サイドバー最下部：免責事項
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    st.caption("【免責事項】")
    st.caption("""
    本ソフトウェアは研究用ツールとして「現状有姿」で提供されます。
    開発者は、本ツールの計算結果の正確性、完全性、特定目的への適合性について一切の保証を行いません。
    本ツールの使用により生じた、いかなる損害についても、開発者は責任を負いません。
    """)
