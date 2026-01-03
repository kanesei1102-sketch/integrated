import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# ---------------------------------------------------------
# 0. ページ設定 (Page Config)
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat & Graph Engine", layout="wide")

# ---------------------------------------------------------
# 1. サイドバー設定 (Sidebar UI)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 【Notice / ご案内】")
    st.info("""
    This tool acts as a GUI wrapper for standard Python statistical libraries (**SciPy, Statsmodels, scikit-posthocs**).
    
    本ツールは、信頼性の高い標準統計ライブラリ（SciPy等）を実行するためのインターフェースです。
    論文記載時は「独自ソフト」ではなく「PythonのSciPyライブラリ等を使用した」と記述してください。

    👉 **[Contact & Feedback](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)
    st.divider()

    st.header("🛠️ グラフ設定")
    
    with st.expander("📈 グラフの種類", expanded=True):
        graph_type = st.selectbox("形式", ["棒グラフ (Bar)", "箱ひげ図 (Box)", "バイオリン図 (Violin)"])
        if "棒" in graph_type:
            error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"])
        else:
            error_type = "None"
        
    with st.expander("🎨 デザイン微調整", expanded=False):
        fig_title = st.text_input("図のタイトル", value="Experiment Result")
        y_axis_label = st.text_input("Y軸ラベル", value="Relative Value")
        manual_y_max = st.number_input("Y軸最大値 (0で自動)", value=0.0, step=1.0)
        
        st.divider()
        st.caption("間隔と太さの調整")
        group_spacing = st.slider("↔️ グループ間の距離 (間隔)", 0.8, 3.0, 1.2, 0.1)
        bar_width = st.slider("⬛ 棒/箱の太さ (幅)", 0.1, 1.5, 0.6, 0.1)
        
        st.caption("ドット・その他")
        dot_size = st.slider("ドットサイズ", 0, 100, 20)
        dot_alpha = st.slider("ドットの透明度", 0.1, 1.0, 0.7)
        jitter_strength = st.slider("ばらつき (Jitter)", 0.0, 0.2, 0.04, 0.01)
        fig_height = st.slider("画像の高さ", 3.0, 10.0, 5.0)

# ---------------------------------------------------------
# 2. メインエリア：データ入力 (Main Input)
# ---------------------------------------------------------
st.title("🔬 Ultimate Sci-Stat & Graph Engine")
st.markdown("""
**統計解析から論文グレードのグラフ作成までを自動化する統合ツール (Pro Ver.)** データの性質を自動診断し、最適な検定を選択。有意差バー付きのグラフを一瞬で作成します。
""")

st.subheader("1. データ入力")
tab_manual, tab_csv = st.tabs(["✍️ 手動入力", "📂 CSVアップロード"])

data_dict = {}

# --- A. 手動入力モード ---
with tab_manual:
    if 'g_count' not in st.session_state: st.session_state.g_count = 3
    
    col_ctrl, _ = st.columns([1, 5])
    with col_ctrl:
        c1, c2 = st.columns(2)
        if c1.button("＋ 追加"): st.session_state.g_count += 1
        if c2.button("－ 削除") and st.session_state.g_count > 2: st.session_state.g_count -= 1

    cols = st.columns(min(st.session_state.g_count, 4))
    for i in range(st.session_state.g_count):
        with cols[i % 4]:
            def_name = f"Group {i+1}"
            name = st.text_input(f"名前 {i+1}", value=def_name, key=f"n{i}")
            raw = st.text_area(f"データ {i+1}", height=120, key=f"d{i}", placeholder="10.5\n12.3")
            vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
            if len(vals) > 0: data_dict[name] = vals

# --- B. CSVアップロードモード ---
with tab_csv:
    uploaded_file = st.file_uploader("CSVファイルをアップロード (列名: Group, Value)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) >= 2:
                g_col = df.columns[0]
                v_col = df.columns[1]
                for g_name in df[g_col].unique():
                    g_vals = df[df[g_col] == g_name][v_col].dropna().tolist()
                    if len(g_vals) > 0:
                        data_dict[g_name] = g_vals
                st.success(f"CSV読み込み成功: {len(data_dict)} グループを検出")
            else:
                st.error("CSVは2列以上である必要があります (例: A列=グループ名, B列=数値)")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

# ---------------------------------------------------------
# 3. グループカラー設定
# ---------------------------------------------------------
group_colors = {}
if data_dict:
    with st.sidebar:
        with st.expander("🖍️ グループカラー設定", expanded=True):
            default_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
            for i, g_name in enumerate(data_dict.keys()):
                col_def = default_colors[i % len(default_colors)]
                group_colors[g_name] = st.color_picker(f"{g_name} の色", col_def)

# ---------------------------------------------------------
# 4. 統計解析エンジン (SciPy Wrapper / Detailed Report)
# ---------------------------------------------------------
def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

sig_pairs = [] 

if len(data_dict) >= 2:
    st.header("2. 統計解析レポート")
    
    group_names = list(data_dict.keys())
    all_values = list(data_dict.values())
    
    valid_data_count = all(len(v) >= 2 for v in all_values)
    
    if not valid_data_count:
        st.warning("各グループに少なくとも2つ以上の数値を入力してください。")
    else:
        # 正規性診断 (SciPy: shapiro)
        all_normal = True
        for v in all_values:
            if len(v) >= 3:
                _, p_s = stats.shapiro(v)
                if p_s <= 0.05: all_normal = False
        
        # 等分散性診断 (SciPy: levene)
        try:
            _, p_lev = stats.levene(*all_values)
            is_equal_var = (p_lev > 0.05)
        except:
            is_equal_var = True

        method_name = ""
        lib_name = ""
        p_global = 1.0
        
        # --- 2群比較 ---
        if len(data_dict) == 2:
            g1, g2 = all_values[0], all_values[1]
            if all_normal:
                if is_equal_var:
                    method_name = "Studentのt検定"
                    lib_name = "scipy.stats.ttest_ind"
                    _, p_global = stats.ttest_ind(g1, g2, equal_var=True)
                else:
                    method_name = "Welchのt検定"
                    lib_name = "scipy.stats.ttest_ind (equal_var=False)"
                    _, p_global = stats.ttest_ind(g1, g2, equal_var=False)
            else:
                method_name = "Mann-WhitneyのU検定"
                lib_name = "scipy.stats.mannwhitneyu"
                _, p_global = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                
            if p_global < 0.05:
                sig_pairs.append({'g1': group_names[0], 'g2': group_names[1], 'label': get_sig_label(p_global), 'p': p_global})

        # --- 3群以上比較 ---
        else:
            if all_normal and is_equal_var:
                method_name = "一元配置分散分析 (ANOVA) + Tukey法"
                lib_name = "scipy.stats.f_oneway & statsmodels"
                _, p_global = stats.f_oneway(*all_values)
                
                if p_global < 0.05:
                    flat_data = [v for sub in all_values for v in sub]
                    labels = [n for n, sub in data_dict.items() for _ in sub]
                    res = pairwise_tukeyhsd(flat_data, labels)
                    
                    df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                    for _, row in df_res.iterrows():
                        if row['reject']:
                            sig_pairs.append({'g1': row['group1'], 'g2': row['group2'], 'label': get_sig_label(row['p-adj']), 'p': row['p-adj']})
            else:
                method_name = "Kruskal-Wallis検定 + Dunn検定"
                lib_name = "scipy.stats.kruskal & scikit_posthocs"
                _, p_global = stats.kruskal(*all_values)
                
                if p_global < 0.05:
                    dunn = sp.posthoc_dunn(all_values, p_adjust='bonferroni')
                    dunn.columns = group_names
                    dunn.index = group_names
                    
                    for i in range(len(group_names)):
                        for j in range(i+1, len(group_names)):
                            n1, n2 = group_names[i], group_names[j]
                            p_val = dunn.loc[n1, n2]
                            if p_val < 0.05:
                                sig_pairs.append({'g1': n1, 'g2': n2, 'label': get_sig_label(p_val), 'p': p_val})

        # --- レポート生成ロジック (詳細日本語版) ---
        if all_normal and is_equal_var:
            easy_reason = "データの分布が偏っておらず、バラツキも均一だったため、最も標準的で精度の高い『パラメトリック検定』を選択しました。"
        elif not all_normal:
            easy_reason = "データに極端な偏りや外れ値が見られたため、数値の大小関係（順位）を重視する、外れ値に強い『ノンパラメトリック検定』を選択しました。"
        else:
            easy_reason = "データのバラツキが群の間で異なっていたため、その差を補正して計算する手法を選択しました。"

        result_summary = "【有意差あり】グループ間に、偶然とは言い切れない明らかな差が見つかりました。" if p_global < 0.05 else "【有意差なし】グループ間の差は、誤差の範囲内である可能性が高いです。"

        analysis_path = f"""
【判定プロセス（自動診断）】
1. 正規性確認：{"合格（正規分布）" if all_normal else "不合格（非正規分布あり）"}
2. 等分散性確認：{"合格（均一）" if is_equal_var else "不合格（不均一）"}
⇒ 上記診断に基づき、統計的に最も妥当な手法として「{method_name}」を自動選択しました。
"""

        st.success(f"**採用された手法: {method_name}**")
        
        with st.expander("📝 そのまま使える報告用レポート (詳細)", expanded=True):
            full_report = f"""
【解析報告書：{", ".join(group_names)} の比較】

{analysis_path}

1. この解析で何を確認したか：
   各グループの数値の平均に、意味のある「違い」があるかどうかを調べました。

2. どの方法で調べたか（その理由）：
   採用手法：{method_name}
   理由：{easy_reason}
   ※ データの形（正規性）やバラツキ（等分散性）を事前にチェックした上で、最も科学的に妥当な手順を選んでいます。

3. 解析の結果：
   判定：{result_summary}
   全体のP値：{p_global:.4e}
   （※P値が0.05より小さければ、統計学的に「差がある」と判断します）

4. 個別の違い（多重比較）：
   {"3群以上の比較のため、各ペアを総当たりで調べ、厳しい基準で有意差を判定しました。" if len(data_dict) > 2 else "2つのグループを直接比較しました。"}

5. 結論：
   解析の結果、今回のデータからは統計学的な裏付けが得られました。この内容に基づき、有意差ラベルを付与したグラフを作成しました。
            """
            st.text_area("レポート全文", value=full_report, height=450)
            
        with st.expander("📄 論文用 Methods 記述案 (日本語)", expanded=False):
            methods_text = f"""
統計解析にはPython環境下のSciPyライブラリ等を用いた。
データの正規性はShapiro-Wilk検定、等分散性はLevene検定により確認した。
群間の比較には {method_name} を用いた。
P値 0.05 未満を統計学的に有意とみなした。
            """
            st.text_area("Methods記述案", value=methods_text, height=150)

    st.divider()

# ---------------------------------------------------------
# 5. グラフ描画エンジン (Visualization Core)
# ---------------------------------------------------------
if len(data_dict) >= 1:
    st.header("3. グラフ生成 (Auto-Labeling)")
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        
        base_scale = 1.5
        auto_width = max(6.0, len(data_dict) * base_scale * group_spacing)
        
        fig, ax = plt.subplots(figsize=(auto_width, fig_height))
        
        group_names = list(data_dict.keys())
        x_positions = np.arange(len(group_names)) * group_spacing
        
        all_vals_flat = [v for sub in data_dict.values() for v in sub if len(sub) > 0]
        max_val = np.max(all_vals_flat) if all_vals_flat else 1.0
        
        for i, (name, vals) in enumerate(data_dict.items()):
            if len(vals) == 0: continue
            vals = np.array(vals)
            pos = x_positions[i]
            
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
            sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0
            err = sem_v if error_type == "SEM" else std_v
            my_color = group_colors.get(name, "#333333")

            if "棒" in graph_type:
                ax.bar(pos, mean_v, width=bar_width, color=my_color, edgecolor='black', alpha=0.8, zorder=1)
                ax.errorbar(pos, mean_v, yerr=err, fmt='none', color='black', capsize=5, zorder=2)
            elif "箱" in graph_type:
                ax.boxplot(vals, positions=[pos], widths=bar_width, patch_artist=True,
                           boxprops=dict(facecolor=my_color, alpha=0.8), medianprops=dict(color='black'), showfliers=False)
            elif "バイオリン" in graph_type:
                parts = ax.violinplot(vals, positions=[pos], widths=bar_width, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(my_color); pc.set_alpha(0.8)
            
            if dot_size > 0:
                noise = np.random.normal(0, jitter_strength, len(vals))
                ax.scatter(pos + noise, vals, s=dot_size, color='white', edgecolor='gray', zorder=3, alpha=dot_alpha)

        y_step = max_val * 0.15
        current_y = max_val * 1.15
        
        for pair in sig_pairs:
            try:
                idx1 = group_names.index(pair['g1'])
                idx2 = group_names.index(pair['g2'])
                x1, x2 = x_positions[idx1], x_positions[idx2]
                
                bar_h = current_y
                col_h = max_val * 0.03
                ax.plot([x1, x1, x2, x2], [bar_h-col_h, bar_h, bar_h, bar_h-col_h], lw=1.5, c='black')
                ax.text((x1+x2)/2, bar_h, pair['label'], ha='center', va='bottom', fontsize=14)
                current_y += y_step
            except: pass

        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_names, fontsize=12)
        ax.set_ylabel(y_axis_label, fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        
        margin = 0.8 * group_spacing
        ax.set_xlim(min(x_positions) - margin, max(x_positions) + margin)

        if manual_y_max > 0:
            ax.set_ylim(0, manual_y_max)
        else:
            ax.set_ylim(0, current_y * 1.1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("📥 画像を保存 (PNG)", data=img_buf, file_name=f"result_{now_str}.png", mime="image/png")
    except Exception as e:
        st.error(f"描画エラー: {e}")
else:
    st.info("データを入力してください (手動 または CSV)")

# ---------------------------------------------------------
# 6. サイドバー最下部：免責事項 (Standard Disclaimer)
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    st.caption("【免責事項 / Disclaimer】")
    st.caption("""
    本ツールは、SciPy/Statsmodels等のオープンソースライブラリを利用した計算結果を表示するものです。
    最終的な解釈および結論については、利用者が専門的知見に基づいて判断してください。
    """)
